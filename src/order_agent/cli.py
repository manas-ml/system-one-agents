"""Command line: run the demo scenarios or chat with the agent.

    python -m order_agent list
    python -m order_agent demo                      # all scenarios, mock System One API
    python -m order_agent demo which-order --live   # real Jev via OpenRouter
    python -m order_agent chat --live               # you are the customer and the operator
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from jevkit import (
    DEFAULT_MODEL,
    CachingTransport,
    Jev,
    OpenRouterTransport,
    Transport,
    TransportError,
)
from jevkit.mock_server import MockSystemOneServer

from .agent import OrderAgent, Outcome
from .offline import offline_jev
from .scenarios import SCENARIOS, Scenario, ScriptedChannel
from .shop import Record, ToolName, seed_db
from .trace import Event, Trace

# --------------------------------------------------------------------------- printing

_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t: str) -> str: return _c("2", t)
def bold(t: str) -> str: return _c("1", t)
def green(t: str) -> str: return _c("32", t)
def red(t: str) -> str: return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)
def blue(t: str) -> str: return _c("34", t)
def magenta(t: str) -> str: return _c("35", t)


def _bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "█" * filled + dim("░" * (width - filled))


def _short(value: Any, limit: int = 110) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in value.items()) if isinstance(value, dict) else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def print_event(event: Event) -> None:
    d = event.data
    match event.kind:
        case "customer":
            print(f"\n  {bold('customer')} › {d['text']}")
        case "agent":
            print(f"\n  {bold(magenta('agent'))}    › {d['text']}")
        case "decision":
            meta = f"{d['model']}"
            if d["latency_ms"] >= 1:
                meta += f" · {d['latency_ms']:.0f} ms"
            print(f"\n  {blue('Jev')} ▸ {bold(d['key'])} {dim('(' + d['kind'] + ')')}"
                  f"  {dim(meta)}")
            probs: dict[str, float] = d["probabilities"]
            if d["kind"] == "noul":
                print(f"      P(true) {_bar(probs['true'])} {probs['true']:.2f}")
                return
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            width = max(len(k) for k, _ in ranked)
            for k, p in ranked[:5]:
                mark = green(" ◀") if k == d["answer"] else ""
                print(f"      {k:<{width}} {_bar(p)} {p:.2f}{mark}")
        case "tool_call":
            print(f"\n  {yellow('→')} {bold(d['tool'])}({_short(d['args'])})")
        case "tool_result":
            print(f"  {green('←')} {dim(_short(d['result'], 140))}")
        case "tool_error":
            print(f"  {red('✗')} {d['error']}")
        case "gate":
            verdict = green("PASS") if d["passed"] else red("BLOCKED")
            checks = "  ".join(f"{k}={p:.2f}" for k, p in d["checks"].items())
            print(f"  {bold('policy gate')} {verdict} {dim('(each ≥ ' + str(d['threshold']) + ')')}"
                  f"  {checks}")
        case "approval":
            verdict = green("approved") if d["approved"] else red("rejected")
            print(f"  {bold('operator')} {verdict} {d['tool']}({_short(d['args'])})")
        case "note":
            print(f"  {dim('· ' + d['text'])}")
        case "outcome":
            colour = {"resolved": green, "informed": green, "escalated": yellow,
                      "declined": yellow}.get(d["status"], red)
            print(f"\n  {bold('outcome')} {colour(d['status'].upper())}: {d['summary']}")


# --------------------------------------------------------------------------- wiring


def make_transport(backend: str, cache_dir: Path | None, stack: ExitStack) -> Transport:
    """mock: real HTTP client -> local mock of OpenRouter's System One API (default).
    offline: rules called in-process. live: Jev on OpenRouter. replay: cache only."""
    inner: Transport | None
    if backend == "replay":
        inner = None
    elif backend == "live":
        inner = OpenRouterTransport.from_env()
    elif backend == "mock":
        server = stack.enter_context(MockSystemOneServer(offline_jev()))
        inner = OpenRouterTransport(api_key="mock-key", base_url=server.base_url)
    else:
        inner = offline_jev()
    if cache_dir is None and backend != "replay":
        assert inner is not None
        return inner
    return CachingTransport(inner=inner, directory=cache_dir or Path(".jev-cache"),
                            mode="replay" if backend == "replay" else "record")


BACKEND_LABEL = {
    "mock": "mock System One API on localhost (OpenRouter wire format; illustrative "
            "probabilities) - use --live for Jev",
    "offline": "offline rules, in-process (illustrative probabilities)",
    "replay": "replay from cache",
}


def run_scenario(s: Scenario, transport: Transport, model: str, quiet: bool = False,
                 trace_dir: Path | None = None) -> tuple[Outcome, Trace]:
    trace = Trace()
    if not quiet:
        trace.listeners.append(print_event)
        print("\n" + bold("━" * 78))
        print(bold(f" {s.id}: {s.title}"))
        print(dim(f" shows: {s.shows}"))
    jev = Jev(transport=transport, model=model, session_id=f"order-agent-{s.id}",
              on_decision=trace.on_decision)
    agent = OrderAgent(jev=jev, db=seed_db(),
                       channel=ScriptedChannel(list(s.replies), approve_writes=s.approve),
                       trace=trace)
    outcome = agent.run(s.message)
    if not quiet:
        print(dim(f"  {jev.calls} Jev requests, {len(trace.decisions())} decisions, "
                  f"{len(outcome.tool_calls)} tool calls"))
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{s.id}.json").write_text(trace.to_json())
    return outcome, trace


class ConsoleChannel:
    def ask_customer(self, question: str) -> str | None:
        try:
            return input(f"  {bold('you')}      › ").strip() or None
        except EOFError:
            return None

    def approve(self, tool: ToolName, args: Record) -> bool:
        try:
            answer = input(f"  {bold('operator')}: run {tool.value}"
                           f"({_short(args.model_dump(mode='json'))})? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="order_agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the demo scenarios")
    for name in ("demo", "chat"):
        p = sub.add_parser(name)
        if name == "demo":
            p.add_argument("scenario", nargs="*", help="scenario ids (default: all)")
            p.add_argument("--trace-dir", type=Path, help="write a JSON audit trail per scenario")
        p.add_argument("--backend", choices=["mock", "offline", "live", "replay"],
                       default=os.environ.get("JEV_BACKEND", "mock"),
                       help="mock (default): local mock of the System One API; "
                            "live: Jev via OpenRouter (needs OPENROUTER_API_KEY)")
        p.add_argument("--live", action="store_const", dest="backend", const="live",
                       help="shorthand for --backend live")
        p.add_argument("--model", default=os.environ.get("JEV_MODEL", DEFAULT_MODEL))
        p.add_argument("--cache-dir", type=Path, help="record answers here for replay")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for s in SCENARIOS:
            print(f"{s.id:<17} {s.title}\n{'':<17} {dim(s.shows)}")
        return 0

    with ExitStack() as stack:
        return _run(args, parser, stack)


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser, stack: ExitStack) -> int:
    try:
        transport = make_transport(args.backend, args.cache_dir, stack)
    except TransportError as err:
        print(red(f"error: {err}. Export your OpenRouter key as OPENROUTER_API_KEY, "
                  "or drop --live to use the mock backend."), file=sys.stderr)
        return 2
    model = args.model
    label = BACKEND_LABEL.get(args.backend, f"Jev via OpenRouter ({model})")
    print(dim(f"backend: {label}"))

    if args.cmd == "chat":
        trace = Trace(listeners=[print_event])
        jev = Jev(transport=transport, model=model, on_decision=trace.on_decision)
        agent = OrderAgent(jev=jev, db=seed_db(), channel=ConsoleChannel(), trace=trace)
        print(dim("Customers: jane@acme.com, omar@example.com, li@example.com, "
                  "sam@example.com, ana@example.com"))
        try:
            message = input(f"  {bold('you')}      › ")
        except EOFError:
            return 0
        agent.run(message)
        return 0

    wanted = set(args.scenario)
    unknown = wanted - {s.id for s in SCENARIOS}
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(sorted(unknown))}")
    failures = 0
    for s in SCENARIOS:
        if wanted and s.id not in wanted:
            continue
        outcome, _ = run_scenario(s, transport, model, trace_dir=args.trace_dir)
        ok = outcome.status == s.expect_status and tuple(outcome.tool_calls) == tuple(
            t.value for t in s.expect_tools)
        if not ok:
            failures += 1
            print(red(f"  ✗ expected {s.expect_status} via "
                      f"{[t.value for t in s.expect_tools]}"))
    return 1 if failures and args.backend != "live" else 0
