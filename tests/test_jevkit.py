import io
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest

from jevkit import CachingTransport, Jev, NotAnswered, OpenRouterTransport, ReplayMiss
from jevkit.wire import DecisionRequest, DecisionResponse


class Color(StrEnum):
    RED = "red"
    BLUE = "blue"


class Canned:
    """Transport that returns a fixed wire response and remembers the request."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.requests: list[DecisionRequest] = []

    def send(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        return DecisionResponse.model_validate(
            {"model": "typesafe/jev-1.14", "answers": self.answers,
             "usage": {"input_tokens": 10, "output_tokens": 0, "cost": 0.00001}})


def test_choose_maps_wire_keys_back_to_typed_values() -> None:
    t = Canned({"c": {"type": "choice", "choice": "blue",
                      "probabilities": {"red": 0.2, "blue": 0.8}}})
    pick = Jev(t).choose({"x": 1}, "c", "Which colour?", {Color.RED: "warm", Color.BLUE: "cool"})
    assert pick.value is Color.BLUE
    assert pick.probabilities == {Color.RED: 0.2, Color.BLUE: 0.8}
    assert pick.margin == pytest.approx(0.6)
    sent = t.requests[0].model_dump(mode="json", exclude_none=True)
    assert sent["questions"]["c"] == {"type": "choice", "instructions": "Which colour?",
                                      "criteria": {"red": "warm", "blue": "cool"}}


@pytest.mark.parametrize("answer", [
    {"type": "refusal"},
    {"type": "noul", "noul": 0.9},                                   # wrong type
    {"type": "choice", "choice": "green", "probabilities": {"green": 1.0}},  # not offered
])
def test_unusable_answers_fail_closed(answer: dict[str, Any]) -> None:
    with pytest.raises(NotAnswered):
        Jev(Canned({"c": answer})).choose("s", "c", "?", {Color.RED: "a", Color.BLUE: "b"})


def test_missing_answer_fails_closed() -> None:
    with pytest.raises(NotAnswered):
        Jev(Canned({})).check("s", "k", "Is it?")


def test_checks_batch_into_one_request() -> None:
    t = Canned({"a": {"type": "noul", "noul": 0.9}, "b": {"type": "noul", "noul": 0.1}})
    out = Jev(t).checks("s", {"a": "A holds", "b": "B holds"})
    assert len(t.requests) == 1
    assert out["a"].holds(0.8) and not out["b"].holds(0.8)


def test_cache_records_then_replays(tmp_path: Path) -> None:
    t = Canned({"k": {"type": "noul", "noul": 0.7}})
    rec = Jev(CachingTransport(t, tmp_path))
    assert rec.check({"a": 1}, "k", "?").probability == 0.7
    replay = Jev(CachingTransport(None, tmp_path, mode="replay"))
    assert replay.check({"a": 1}, "k", "?").probability == 0.7
    assert len(t.requests) == 1
    with pytest.raises(ReplayMiss):
        replay.check({"a": 2}, "k", "?")


def test_openrouter_transport_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Resp(io.BytesIO):
        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *a: object) -> None:
            pass

    def fake_urlopen(req: Any, timeout: float) -> Resp:
        seen["url"], seen["auth"] = req.full_url, req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return Resp(json.dumps({"id": "gen-dec-1", "model": "typesafe/jev-1.14-20261014",
                                "provider": "TypeSafe",
                                "answers": {"k": {"type": "noul", "noul": 0.98}},
                                "usage": {"input_tokens": 5, "output_tokens": 0,
                                          "cost": 0.0000021}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    jev = Jev(OpenRouterTransport(api_key="sk-or-test"))
    assert jev.check("I was charged twice.", "k", "Asks for money back?").probability == 0.98
    assert seen["url"] == "https://openrouter.ai/api/v1/systemone"
    assert seen["auth"] == "Bearer sk-or-test"
    assert seen["body"] == {"model": "typesafe/jev-1.14", "state": "I was charged twice.",
                            "questions": {"k": {"type": "noul",
                                                "instructions": "Asks for money back?"}}}
