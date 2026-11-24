import json
import urllib.error
import urllib.request

import pytest

from jevkit import Jev, OpenRouterTransport, TransportError
from jevkit.mock_server import MockSystemOneServer
from order_agent.agent import OrderAgent
from order_agent.offline import offline_jev
from order_agent.scenarios import SCENARIOS, ScriptedChannel
from order_agent.shop import seed_db
from order_agent.trace import Trace


def test_full_scenarios_over_http() -> None:
    """The real HTTP transport, end to end, against the local mock API."""
    with MockSystemOneServer(offline_jev()) as server:
        transport = OpenRouterTransport(api_key="mock-key", base_url=server.base_url)
        for s in SCENARIOS:
            trace = Trace()
            agent = OrderAgent(jev=Jev(transport, on_decision=trace.on_decision), db=seed_db(),
                               channel=ScriptedChannel(list(s.replies), s.approve), trace=trace)
            outcome = agent.run(s.message)
            assert outcome.status == s.expect_status, s.id
            assert trace.tool_calls() == [t.value for t in s.expect_tools], s.id
        assert server.requests and all(r.model == "typesafe/jev-1.14" for r in server.requests)


def test_mock_rejects_missing_auth_and_bad_paths() -> None:
    with MockSystemOneServer(offline_jev()) as server:
        body = json.dumps({"model": "m", "state": "s",
                           "questions": {"k": {"type": "noul", "instructions": "?"}}}).encode()
        for path, headers, code in [("/v1/systemone", {}, 401), ("/v1/other", {"Authorization": "Bearer x"}, 404)]:
            req = urllib.request.Request(server.base_url + path, data=body, method="POST",
                                         headers={"Content-Type": "application/json", **headers})
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(req)
            assert err.value.code == code


def test_transport_surfaces_4xx_without_retry() -> None:
    with MockSystemOneServer(offline_jev()) as server:
        bad = OpenRouterTransport(api_key="k", base_url=server.base_url, path="/v1/nope")
        with pytest.raises(TransportError, match="404"):
            Jev(bad).check("s", "in_scope", "?")
