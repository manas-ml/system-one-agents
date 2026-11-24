"""Offline stand-in for Jev, so the demo and tests run with no network and no key.

These hand-written rules answer the agent's questions in Jev's exact wire format.
They are not Jev: the probabilities are illustrative. Run with `--live` to get
Jev's real answers through OpenRouter. The agent code path is the same either way.
"""

from __future__ import annotations

import re
from typing import Any

from jevkit import OfflineTransport
from jevkit.wire import ChoiceQuestion, NoulQuestion

State = dict[str, Any]

_WORD = re.compile(r"[a-z]+")
_STOP = {"the", "a", "an", "my", "of", "it", "and", "to", "i", "is", "placed", "total"}


def _customer_text(state: State) -> str:
    return " ".join(m["text"] for m in state.get("conversation", [])
                    if m.get("from") == "customer").lower()


def _words(text: str) -> set[str]:
    return {w.rstrip("s") for w in _WORD.findall(text.lower()) if w not in _STOP}


def _has(text: str, *needles: str) -> bool:
    return any(n in text for n in needles)


def _last(state: State, tool: str) -> dict[str, Any] | None:
    for h in reversed(state.get("history", [])):
        if h["tool"] == tool and "result" in h:
            result: dict[str, Any] = h["result"]
            return result
    return None


def _shipment_status(state: State) -> str | None:
    shipment = _last(state, "track_shipment")
    return None if shipment is None else str(shipment["status"])


def _wants_replacement(text: str) -> bool:
    return _has(text, "resend", "replace", "send it again", "send another", "ship it again")


def _wants_refund(text: str) -> bool:
    return _has(text, "refund", "money back", "reimburse")


def _injection(text: str) -> bool:
    return _has(text, "note to agent", "ignore", "policy was updated", "skip any checks")


def _order_overlap(state: State, description: str) -> int:
    items = description.split(":", 1)[-1].split(";", 1)[0]
    return len(_words(items) & _words(_customer_text(state)))


# --------------------------------------------------------------------------- noul


def in_scope(state: State, _q: NoulQuestion) -> float:
    text = _customer_text(state)
    return 0.93 if _has(text, "order", "refund", "parcel", "package", "arriv", "deliver",
                        "resend", "lost", "where are my", "never showed") else 0.04


def express(state: State, _q: NoulQuestion) -> float:
    return 0.91 if _has(_customer_text(state), "fast", "express", "asap", "urgent",
                        "quickly") else 0.08


def customer_asked_refund(state: State, _q: NoulQuestion) -> float:
    return 0.95 if _wants_refund(_customer_text(state)) else 0.12


def customer_asked_replacement(state: State, _q: NoulQuestion) -> float:
    return 0.93 if _wants_replacement(_customer_text(state)) else 0.1


def right_order(state: State, _q: NoulQuestion) -> float:
    orders: dict[str, str] = state.get("orders", {})
    proposed = state.get("proposed", {}).get("order_id")
    if len(orders) <= 1:
        return 0.94
    scores = {k: _order_overlap(state, v) for k, v in orders.items()}
    return 0.92 if scores.get(proposed, 0) == max(scores.values()) > 0 else 0.2


def policy_allows(state: State, _q: NoulQuestion) -> float:
    # Lost parcels qualify for a refund or a replacement; delivered or in-transit ones don't.
    return {"lost": 0.92, "delivered": 0.05, "in_transit": 0.07}.get(
        _shipment_status(state) or "", 0.1)


# --------------------------------------------------------------------------- choice


def next_step(state: State, q: ChoiceQuestion) -> dict[str, float]:
    opts = q.criteria
    text = _customer_text(state)
    w = {k: 0.03 for k in opts}

    def pick(key: str, p: float) -> None:
        w[key] = p

    if _last(state, "send_email") is not None:
        pick("finish", 0.92)
    elif "find_customer" in opts:
        pick("find_customer", 0.9)
    elif "list_orders" in opts:
        pick("list_orders", 0.9)
    elif "track_shipment" in opts:
        pick("track_shipment", 0.88)
    elif "issue_refund" in opts:  # shipment known, nothing done yet
        status = _shipment_status(state)
        if status == "lost":
            pick("reship_order" if _wants_replacement(text) else "issue_refund", 0.85)
        elif status == "delivered":
            if _injection(text):  # deliberately fooled, to show the policy gate catching it
                pick("issue_refund", 0.62)
                pick("escalate_to_human", 0.3)
            else:
                pick("escalate_to_human", 0.8)
                pick("issue_refund", 0.1)
        else:
            pick("send_email", 0.8)
            pick("issue_refund", 0.09)
    elif "send_email" in opts:
        pick("send_email", 0.9)
    else:
        pick("finish", 0.92)
    return w


def order_id(state: State, q: ChoiceQuestion) -> dict[str, float]:
    w = {k: 1.0 + 3 * _order_overlap(state, v) for k, v in q.criteria.items() if k != "none"}
    w["none"] = 0.08
    return w


def refund_reason(state: State, q: ChoiceQuestion) -> dict[str, float]:
    w = {k: 0.03 for k in q.criteria}
    status = _shipment_status(state)
    w["lost_in_transit" if status == "lost" else "late_delivery"] = 0.9
    return w


def refund_scope(state: State, q: ChoiceQuestion) -> dict[str, float]:
    shipping = _has(_customer_text(state), "shipping fee", "delivery fee")
    return {"full_order": 0.1 if shipping else 0.88, "shipping_fee": 0.85 if shipping else 0.12}


def escalation_reason(state: State, q: ChoiceQuestion) -> dict[str, float]:
    w = {k: 0.03 for k in q.criteria}
    w["carrier_says_delivered" if _shipment_status(state) == "delivered" else "other"] = 0.85
    return w


def email(state: State, q: ChoiceQuestion) -> dict[str, float]:
    keys = list(q.criteria)
    return {k: (0.8 if i == 0 else 0.2) for i, k in enumerate(keys)}


def offline_jev() -> OfflineTransport:
    return OfflineTransport(
        noul_rules={
            "in_scope": in_scope,
            "express": express,
            "customer_asked_refund": customer_asked_refund,
            "customer_asked_replacement": customer_asked_replacement,
            "right_order": right_order,
            "policy_allows": policy_allows,
        },
        choice_rules={
            "next_step": next_step,
            "order_id": order_id,
            "refund_reason": refund_reason,
            "refund_scope": refund_scope,
            "escalation_reason": escalation_reason,
            "email": email,
        },
    )
