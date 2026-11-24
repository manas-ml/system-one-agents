import re
from datetime import timedelta

import pytest

from jevkit import Jev, OfflineTransport
from order_agent.agent import OrderAgent
from order_agent.offline import offline_jev
from order_agent.scenarios import SCENARIOS, Scenario, ScriptedChannel
from order_agent.shop import TODAY, seed_db
from order_agent.trace import Trace


def run(s: Scenario, transport: object = None) -> tuple[object, Trace, ScriptedChannel]:
    trace = Trace()
    channel = ScriptedChannel(list(s.replies), approve_writes=s.approve)
    jev = Jev(transport or offline_jev(), on_decision=trace.on_decision)  # type: ignore[arg-type]
    agent = OrderAgent(jev=jev, db=seed_db(), channel=channel, trace=trace)
    return agent.run(s.message), trace, channel


@pytest.mark.parametrize("s", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario_chain(s: Scenario) -> None:
    outcome, trace, _ = run(s)
    assert outcome.status == s.expect_status  # type: ignore[attr-defined]
    assert trace.tool_calls() == [t.value for t in s.expect_tools]


@pytest.mark.parametrize("s", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_no_invented_values(s: Scenario) -> None:
    """Every ID and amount the agent passes to a tool exists in the data or earlier results."""
    _, trace, _ = run(s)
    db = seed_db()
    seen_text = s.message + " " + " ".join(s.replies)
    produced: set[str] = set()
    for e in trace.events:
        if e.kind == "tool_result":
            produced |= set(re.findall(r"[A-Z]+-\d+", str(e.data["result"])))
        if e.kind != "tool_call":
            continue
        args = e.data["args"]
        if "email" in args:
            assert args["email"] in seen_text
        for key in ("customer_id", "order_id"):
            if args.get(key):
                assert args[key] in produced, f"{key}={args[key]} came from nowhere"
        if e.data["tool"] == "issue_refund":
            order = db.orders[args["order_id"]]
            assert args["amount_cents"] in (order.total_cents, order.shipping_fee_cents)


def test_asks_which_order_when_ambiguous() -> None:
    s = next(x for x in SCENARIOS if x.id == "which-order")
    _, trace, channel = run(s)
    assert len(channel.asked) == 1 and "ORD-4001" in channel.asked[0]
    assert "ORD-4002" in channel.asked[0]


def test_refusals_fail_closed_to_a_human() -> None:
    """If Jev answers nothing usable, the agent escalates instead of acting."""
    s = SCENARIOS[0]
    silent = OfflineTransport(noul_rules={"in_scope": lambda st, q: 0.9}, choice_rules={})
    outcome, trace, _ = run(s, silent)
    assert outcome.status == "escalated"  # type: ignore[attr-defined]
    assert trace.tool_calls() == ["escalate_to_human"]


def test_reship_eta_is_tomorrow_when_express() -> None:
    s = next(x for x in SCENARIOS if x.id == "reship-express")
    _, trace, _ = run(s)
    result = next(e.data["result"] for e in trace.events
                  if e.kind == "tool_result" and e.data["tool"] == "reship_order")
    assert result["express"] is True and result["eta"] == str(TODAY + timedelta(days=1))
