"""Local identity/callback composition; not Foundry per-user ACL proof."""

from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, Mock

import pytest
from azure.ai.agentserver import core
from azure.ai.agentserver.core import FoundryAgentRequestContext

from native.agent import hosted_identity
from native.config import ConfigView
from native.tools import ToolIdentity, compose_tools


@pytest.mark.parametrize("host_identity", ["hosted", "durable-default"])
async def test_host_context_is_not_delegated_search_authorization(
    monkeypatch: pytest.MonkeyPatch, host_identity: str
) -> None:
    # Use the SDK context, not a synthetic Search token or replacement identity
    # provider. Host authentication/context is not Search-audience delegation.
    context = FoundryAgentRequestContext(
        user_id="opaque-platform-user", call_id="opaque-platform-call"
    )
    monkeypatch.setattr(core, "get_request_context", lambda: context)
    identity = hosted_identity() if host_identity == "hosted" else ToolIdentity()
    assert identity.call_id == (
        "opaque-platform-call" if host_identity == "hosted" else None
    )
    assert identity.search_token is None

    credential = Mock(side_effect=AssertionError("No service-identity fallback"))
    embeddings = AsyncMock(side_effect=AssertionError("No embedding before delegation"))
    search = AsyncMock(
        side_effect=AssertionError("No Search request without delegation")
    )
    monkeypatch.setattr("azure.identity.aio.DefaultAzureCredential", credential)
    monkeypatch.setattr("native.tools.Embeddings.get_embeddings", embeddings)
    monkeypatch.setattr("native.tools.MetadataSearch.search", search)
    async with AsyncExitStack() as resources:
        tools = await compose_tools(
            ConfigView({"MULTIMODAL_ENABLED": True}), identity, resources
        )
        assert [tool.name for tool in tools] == ["multimodal_retrieve"]
        # Identity cannot be supplied by a model argument. Invoke the actual
        # composed, audited callback with the same boundary as native tool tests.
        assert set(tools[0].input_model.model_json_schema()["properties"]) == {"input"}
        with pytest.raises(ValueError, match="invocation-local delegated Search"):
            await tools[0].invoke(
                arguments={"input": "Synthetic protected question"}, skip_parsing=True
            )
    credential.assert_not_called()
    embeddings.assert_not_awaited()
    search.assert_not_awaited()
