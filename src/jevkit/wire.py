"""Wire format of OpenRouter's System One API (`POST /api/v1/systemone`).

These models mirror the request and response shapes documented at
https://openrouter.ai/docs/guides/community/typesafe-sdk and
https://openrouter.ai/docs/guides/features/decisions .

Every response is validated against them, so a malformed or unexpected answer
fails loudly instead of flowing into agent logic.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

QuestionKey = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
"""Key format accepted by every Decisions model on OpenRouter."""

State = Union[str, dict[str, Any], list[Any]]
"""Application state: plain text, a JSON object, or a list of text/media parts."""


# --------------------------------------------------------------------------- questions


class NoulQuestion(BaseModel):
    """Binary question: answered with the probability that the statement holds."""

    model_config = ConfigDict(frozen=True)

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: dict[Literal["true", "false"], str] | None = None


class ChoiceQuestion(BaseModel):
    """Pick one option: answered with a probability for each option key."""

    model_config = ConfigDict(frozen=True)

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str] = Field(min_length=2)


class ScoreQuestion(BaseModel):
    """Ordered scale: answered with an expected position on 0..N-1."""

    model_config = ConfigDict(frozen=True)

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str] = Field(min_length=2)


Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")
]


# --------------------------------------------------------------------------- answers


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(ge=0.0, le=1.0)
    confidence: float | None = None


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


class RefusalAnswer(BaseModel):
    type: Literal["refusal"]


Answer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer, RefusalAnswer],
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- envelopes


class DecisionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    state: State
    questions: dict[QuestionKey, Question] = Field(min_length=1)
    session_id: str | None = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: float = 0
    output_tokens: float = 0
    cost: float | None = None


class DecisionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    answers: dict[str, Answer]
    usage: Usage | None = None
    id: str | None = None
    provider: str | None = None
