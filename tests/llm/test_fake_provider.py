import math

from app.llm.base import EmbedTask, LLMProvider, LLMResponse, ToolCall, ToolSpec, UserMessage
from app.llm.fake import FakeProvider, hash_embedding

TOOL = ToolSpec(name="lookup_reference", description="Look up a verse", parameters={})


def test_fake_provider_satisfies_the_protocol() -> None:
    """Protects interface parity; rejects a fake that drifted from the LLMProvider protocol."""
    assert isinstance(FakeProvider(), LLMProvider)


async def test_embeddings_are_deterministic_across_instances() -> None:
    """Protects reproducible tests; rejects embeddings that change between runs or instances."""
    texts = ["The fear of the LORD is the beginning of wisdom.", "Vanity of vanities"]

    first = await FakeProvider().embed(texts, EmbedTask.DOCUMENT)
    second = await FakeProvider().embed(texts, EmbedTask.DOCUMENT)

    assert first == second
    assert first[0] != first[1]


def test_embeddings_have_requested_dimensions_and_unit_norm() -> None:
    """Protects vector shape; rejects wrong-length or unnormalized vectors (break cosine search)."""
    vector = hash_embedding("Proverbs 3:5", EmbedTask.QUERY, 768)

    assert len(vector) == 768
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)


def test_embeddings_depend_on_task() -> None:
    """Protects the document/query split; rejects ignoring `task` like a real provider wouldn't."""
    assert hash_embedding("wisdom", EmbedTask.QUERY, 16) != hash_embedding(
        "wisdom", EmbedTask.DOCUMENT, 16
    )


async def test_scripted_responses_are_returned_in_order_then_default() -> None:
    """Protects scripted answers; rejects reordering them or failing once the script runs out."""
    fake = FakeProvider(responses=["first", "second"])
    messages = [UserMessage("Who wrote Proverbs?")]

    assert await fake.generate(messages) == "first"
    assert await fake.generate(messages) == "second"
    assert await fake.generate(messages) == "Fake answer to: Who wrote Proverbs?"


async def test_stream_reassembles_to_the_same_text_as_generate() -> None:
    """Protects streaming parity; rejects fragments that lose or add characters."""
    fake = FakeProvider(responses=["Trust in the LORD with all your heart"])

    fragments = [f async for f in fake.stream([UserMessage("q")])]

    assert len(fragments) > 1
    assert "".join(fragments) == "Trust in the LORD with all your heart"


async def test_scripted_tool_calls_are_returned_and_calls_recorded() -> None:
    """Protects agent tests; rejects dropping scripted tool calls or not recording the tools."""
    scripted = LLMResponse(
        text="",
        tool_calls=(ToolCall(id="c1", name="lookup_reference", arguments={"ref": "Prov 3:5"}),),
    )
    fake = FakeProvider(tool_responses=[scripted])

    response = await fake.generate_with_tools([UserMessage("q")], "system", [TOOL])

    assert response.tool_calls == scripted.tool_calls
    assert fake.calls[-1].method == "generate_with_tools"
    assert fake.calls[-1].tools == (TOOL,)
