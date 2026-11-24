"""Typed client for Jev, TypeSafe's System One decision model.

The agent never parses text. It asks typed questions and gets typed answers:

    jev.choose(state, "next_action", "...", options={Tool.FIND: "...", ...})
        -> Choice[Tool]            # value is a Tool, probabilities keyed by Tool

    jev.check(state, "customer_asked", "...")
        -> Check                   # probability that the statement holds

    jev.checks(state, {"a": "...", "b": "..."})
        -> dict[str, Check]        # several checks, one request

Anything that is not a clean answer of the expected type (a refusal, a missing
key, an option Jev was not offered) raises `NotAnswered`. Callers fail closed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from .wire import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    RefusalAnswer,
    State,
)

DEFAULT_MODEL = "typesafe/jev-1.14"

K = TypeVar("K", bound=str)


class NotAnswered(RuntimeError):
    """Jev refused, skipped, or answered a question with the wrong type."""


class Transport(Protocol):
    """Anything that can turn a DecisionRequest into a DecisionResponse."""

    def send(self, request: DecisionRequest) -> DecisionResponse: ...


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class Choice(Generic[K]):
    """Answer to a choice question, typed by the option key type."""

    key: str
    value: K
    probabilities: Mapping[K, float]

    @property
    def probability(self) -> float:
        return self.probabilities[self.value]

    def ranked(self) -> list[tuple[K, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)

    @property
    def margin(self) -> float:
        """Gap between the winner and the runner-up (1.0 if there is no runner-up)."""
        ranked = self.ranked()
        return ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0

    def accept(self, min_probability: float, min_margin: float = 0.0) -> bool:
        return self.probability >= min_probability and self.margin >= min_margin


@dataclass(frozen=True)
class Check:
    """Answer to a noul question: P(statement holds)."""

    key: str
    probability: float

    def holds(self, threshold: float) -> bool:
        return self.probability >= threshold


@dataclass(frozen=True)
class DecisionRecord:
    """One answered question, for the audit trail."""

    key: str
    kind: str
    instructions: str
    options: dict[str, str]
    probabilities: dict[str, float]
    answer: str
    model: str
    latency_ms: float
    cost: float | None


@dataclass
class Jev:
    """Typed facade over a System One transport (OpenRouter, cache, or offline fake)."""

    transport: Transport
    model: str = DEFAULT_MODEL
    session_id: str | None = None
    on_decision: Callable[[DecisionRecord], None] | None = None
    calls: int = field(default=0, init=False)

    # ---- raw ------------------------------------------------------------------

    def ask(self, state: State, questions: Mapping[str, Question]) -> dict[str, Answer]:
        """Send one request with several questions; validate that each was answered."""
        request = DecisionRequest(
            model=self.model, state=state, questions=dict(questions), session_id=self.session_id
        )
        started = time.perf_counter()
        response = self.transport.send(request)
        latency_ms = (time.perf_counter() - started) * 1000
        self.calls += 1

        answers: dict[str, Answer] = {}
        for index, (key, question) in enumerate(questions.items()):
            answer = response.answers.get(key)
            if answer is None or isinstance(answer, RefusalAnswer):
                raise NotAnswered(f"Jev did not answer {key!r} ({'refusal' if answer else 'missing'})")
            if answer.type != question.type:
                raise NotAnswered(f"{key!r}: expected {question.type}, got {answer.type}")
            answers[key] = answer
            # Cost and latency belong to the whole request: report them on its first question.
            self._record(key, question, answer, response, latency_ms if index == 0 else 0.0,
                         first=index == 0)
        return answers

    # ---- typed helpers ----------------------------------------------------------

    def choose(
        self, state: State, key: str, instructions: str, options: Mapping[K, str]
    ) -> Choice[K]:
        """Ask Jev to pick one of `options` (keys are the typed values, values describe them)."""
        if len(options) < 2:
            raise ValueError(f"{key!r}: a choice needs at least two options")
        by_wire = {str(option): option for option in options}
        question = ChoiceQuestion(
            instructions=instructions, criteria={str(k): v for k, v in options.items()}
        )
        answer = self.ask(state, {key: question})[key]
        assert isinstance(answer, ChoiceAnswer)
        if answer.choice not in by_wire:
            raise NotAnswered(f"{key!r}: Jev picked {answer.choice!r}, which was not offered")
        raw = answer.probabilities or {answer.choice: answer.confidence or 1.0}
        unknown = set(raw) - set(by_wire)
        if unknown:
            raise NotAnswered(f"{key!r}: probabilities for unknown options {sorted(unknown)}")
        probabilities = {by_wire[w]: float(raw.get(w, 0.0)) for w in by_wire}
        return Choice(key=key, value=by_wire[answer.choice], probabilities=probabilities)

    def check(
        self,
        state: State,
        key: str,
        instructions: str,
        *,
        true: str | None = None,
        false: str | None = None,
    ) -> Check:
        """Ask Jev how likely a statement about `state` is to hold."""
        criteria = {"true": true, "false": false} if true and false else None
        question = NoulQuestion(instructions=instructions, criteria=criteria)
        answer = self.ask(state, {key: question})[key]
        assert isinstance(answer, NoulAnswer)
        return Check(key=key, probability=answer.noul)

    def checks(self, state: State, statements: Mapping[str, str]) -> dict[str, Check]:
        """Several independent noul checks in a single request."""
        questions: dict[str, Question] = {
            k: NoulQuestion(instructions=v) for k, v in statements.items()
        }
        answers = self.ask(state, questions)
        result: dict[str, Check] = {}
        for k, answer in answers.items():
            assert isinstance(answer, NoulAnswer)
            result[k] = Check(key=k, probability=answer.noul)
        return result

    # ---- audit ----------------------------------------------------------------

    def _record(
        self,
        key: str,
        question: Question,
        answer: Answer,
        response: DecisionResponse,
        latency_ms: float,
        first: bool,
    ) -> None:
        if self.on_decision is None:
            return
        if isinstance(answer, NoulAnswer):
            options = {"true": "holds", "false": "does not hold"}
            probabilities = {"true": answer.noul, "false": 1 - answer.noul}
            label = f"{answer.noul:.2f}"
        elif isinstance(answer, ChoiceAnswer):
            assert isinstance(question, ChoiceQuestion)
            options = dict(question.criteria)
            probabilities = dict(answer.probabilities or {answer.choice: 1.0})
            label = answer.choice
        else:  # score
            options = {}
            probabilities = dict(getattr(answer, "probabilities", None) or {})
            label = f"{getattr(answer, 'score', '?')}"
        self.on_decision(
            DecisionRecord(
                key=key,
                kind=question.type,
                instructions=question.instructions,
                options=options,
                probabilities=probabilities,
                answer=label,
                model=response.model,
                latency_ms=latency_ms,
                cost=response.usage.cost if (response.usage and first) else None,
            )
        )
