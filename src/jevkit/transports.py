"""Transports: how a DecisionRequest reaches a System One model.

- `OpenRouterTransport`: the real thing, `POST https://openrouter.ai/api/v1/systemone`.
- `CachingTransport`: wraps another transport; answers are stored under a SHA-256
  of the request, so a run can be replayed exactly and offline.
- `OfflineTransport`: a rule-based stand-in so demos and tests run without a key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .client import Transport
from .wire import ChoiceQuestion, DecisionRequest, DecisionResponse, NoulQuestion, Question

OPENROUTER_BASE_URL = "https://openrouter.ai/api"


class TransportError(RuntimeError):
    pass


@dataclass
class OpenRouterTransport:
    """Calls Jev through OpenRouter's System One API with an OpenRouter API key."""

    api_key: str
    base_url: str = OPENROUTER_BASE_URL
    path: str = "/v1/systemone"
    timeout_s: float = 30.0
    retries: int = 2

    @classmethod
    def from_env(cls) -> OpenRouterTransport:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise TransportError("OPENROUTER_API_KEY is not set")
        return cls(api_key=key, base_url=os.environ.get("JEV_BASE_URL", OPENROUTER_BASE_URL))

    def send(self, request: DecisionRequest) -> DecisionResponse:
        body = request.model_dump_json(exclude_none=True).encode()
        url = self.base_url.rstrip("/") + self.path
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return DecisionResponse.model_validate_json(resp.read())
            except urllib.error.HTTPError as err:
                detail = err.read().decode(errors="replace")[:500]
                last_error = TransportError(f"System One API {err.code}: {detail}")
                if err.code != 429 and err.code < 500:
                    raise last_error from err
            except (urllib.error.URLError, TimeoutError) as err:
                last_error = TransportError(f"System One API unreachable: {err}")
            time.sleep(0.5 * 2**attempt)
        assert last_error is not None
        raise last_error


def request_fingerprint(request: DecisionRequest) -> str:
    canonical = json.dumps(request.model_dump(mode="json", exclude={"session_id"}), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ReplayMiss(TransportError):
    pass


@dataclass
class CachingTransport:
    """Record answers on first use; replay them byte-for-byte afterwards."""

    inner: Transport | None
    directory: Path
    mode: Literal["record", "replay"] = "record"

    def send(self, request: DecisionRequest) -> DecisionResponse:
        path = self.directory / f"{request_fingerprint(request)}.json"
        if path.exists():
            return DecisionResponse.model_validate_json(path.read_text())
        if self.mode == "replay" or self.inner is None:
            raise ReplayMiss(f"no recorded answer for this request ({path.name})")
        response = self.inner.send(request)
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(response.model_dump_json(indent=2))
        return response


# --------------------------------------------------------------------------- offline

NoulRule = Callable[[dict[str, Any], NoulQuestion], float]
ChoiceRule = Callable[[dict[str, Any], ChoiceQuestion], dict[str, float]]


@dataclass
class OfflineTransport:
    """Answers questions with hand-written rules, in the exact wire format Jev uses.

    It is a stand-in so the demo runs without network or key: the probabilities are
    illustrative, not Jev's. The agent code path is identical to the live one.
    """

    noul_rules: dict[str, NoulRule]
    choice_rules: dict[str, ChoiceRule]
    model: str = "offline/jev-stand-in"

    def send(self, request: DecisionRequest) -> DecisionResponse:
        state = request.state if isinstance(request.state, dict) else {"text": request.state}
        answers: dict[str, Any] = {}
        for key, question in request.questions.items():
            answers[key] = self._answer(key, question, state)
        return DecisionResponse.model_validate(
            {"model": self.model, "provider": "offline", "answers": answers,
             "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}}
        )

    def _answer(self, key: str, question: Question, state: dict[str, Any]) -> dict[str, Any]:
        if isinstance(question, NoulQuestion):
            rule = self.noul_rules.get(key)
            if rule is None:
                return {"type": "refusal"}
            p = min(max(rule(state, question), 0.0), 1.0)
            return {"type": "noul", "noul": round(p, 4), "confidence": round(max(p, 1 - p), 4)}
        if isinstance(question, ChoiceQuestion):
            choice_rule = self.choice_rules.get(key)
            if choice_rule is None:
                return {"type": "refusal"}
            raw = choice_rule(state, question)
            weights = {k: max(raw.get(k, 0.0), 0.0) for k in question.criteria}
            total = sum(weights.values()) or 1.0
            probs = {k: round(v / total, 4) for k, v in weights.items()}
            best = max(probs, key=lambda k: probs[k])
            return {"type": "choice", "choice": best, "probabilities": probs,
                    "confidence": probs[best]}
        return {"type": "refusal"}
