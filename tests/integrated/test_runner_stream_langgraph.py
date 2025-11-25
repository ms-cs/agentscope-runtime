# -*- coding: utf-8 -*-
"""Integration test for LangGraph runner with streaming output."""
import os
import pytest
from unittest.mock import AsyncMock, Mock

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda

from agentscope_runtime.engine.schemas.agent_schemas import (
    AgentRequest,
    MessageType,
    RunStatus,
)
from agentscope_runtime.engine.runner import Runner
from agentscope_runtime.adapters.langgraph.memory import (
    LangGraphSessionHistoryMemory,
)
from agentscope_runtime.engine.services.agent_state import (
    InMemoryStateService,
)
from agentscope_runtime.engine.services.session_history import (
    InMemorySessionHistoryService,
)


class MyLangGraphRunner(Runner):
    def __init__(self) -> None:
        super().__init__()
        self.framework_type = "langgraph"

    async def query_handler(
        self,
        msgs,
        request: AgentRequest = None,
        **kwargs,
    ):
        """
        Handle LangGraph agent query.
        """
        session_id = request.session_id
        user_id = request.user_id

        state = await self.state_service.export_state(
            session_id=session_id,
            user_id=user_id,
        )

        # Create a simple LangGraph-like chain for testing
        def simple_response(inputs):
            return AIMessage(content=f"Echo: {inputs['input']}")

        chain = RunnableLambda(simple_response)

        # Convert messages to LangGraph format if needed
        if hasattr(msgs, "__iter__") and not isinstance(msgs, str):
            input_text = " ".join(
                [str(getattr(msg, "content", msg)) for msg in msgs],
            )
        else:
            input_text = str(msgs)

        # Run the chain
        response = chain.invoke({"input": input_text})

        # Yield the response with last flag
        yield response, True

        # Save state if needed
        # For this simple test, we don't need to save complex state


@pytest.mark.asyncio
async def test_langgraph_runner_stream():
    """Test LangGraph runner with streaming output."""
    # Setup services
    state_service = InMemoryStateService()
    session_service = InMemorySessionHistoryService()

    await state_service.start()
    await session_service.start()

    # Create runner
    runner = MyLangGraphRunner()
    runner.state_service = state_service
    runner.session_service = session_service

    # Create request
    request = AgentRequest(
        input=[
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ],
        user_id="test_user",
        session_id="test_session",
    )

    # Run the agent
    response_count = 0
    async for response, is_last in runner.run(request):
        response_count += 1

        # Verify response structure
        assert isinstance(response, AIMessage)
        assert "Echo:" in response.content
        assert isinstance(is_last, bool)

        # For this simple test, we expect only one response
        if response_count > 1:
            pytest.fail("Expected only one response")

    # Verify we got exactly one response
    assert response_count == 1

    # Cleanup
    await state_service.stop()
    await session_service.stop()


@pytest.mark.asyncio
async def test_langgraph_runner_with_memory():
    """Test LangGraph runner with memory integration."""
    # Setup services
    state_service = InMemoryStateService()
    session_service = InMemorySessionHistoryService()

    await state_service.start()
    await session_service.start()

    # Create memory
    memory = LangGraphSessionHistoryMemory(
        service=session_service,
        user_id="test_user",
        session_id="test_session",
    )

    # Add initial message to memory
    initial_message = HumanMessage(content="Previous conversation")
    await memory.add(initial_message)

    # Verify memory content
    memory_content = await memory.get_memory()
    assert len(memory_content) == 1
    assert isinstance(memory_content[0], HumanMessage)
    assert memory_content[0].content == "Previous conversation"

    # Cleanup
    await state_service.stop()
    await session_service.stop()


@pytest.mark.asyncio
async def test_langgraph_runner_multi_turn():
    """Test LangGraph runner with multi-turn conversation."""
    # Setup services
    state_service = InMemoryStateService()
    session_service = InMemorySessionHistoryService()

    await state_service.start()
    await session_service.start()

    # Create runner
    runner = MyLangGraphRunner()
    runner.state_service = state_service
    runner.session_service = session_service

    # First turn
    request1 = AgentRequest(
        input=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "First message"}],
            },
        ],
        user_id="test_user",
        session_id="test_session",
    )

    responses_first = []
    async for response, is_last in runner.run(request1):
        responses_first.append((response, is_last))

    assert len(responses_first) == 1
    assert isinstance(responses_first[0][0], AIMessage)
    assert "Echo: First message" in responses_first[0][0].content

    # Second turn
    request2 = AgentRequest(
        input=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Second message"}],
            },
        ],
        user_id="test_user",
        session_id="test_session",
    )

    responses_second = []
    async for response, is_last in runner.run(request2):
        responses_second.append((response, is_last))

    assert len(responses_second) == 1
    assert isinstance(responses_second[0][0], AIMessage)
    assert "Echo: Second message" in responses_second[0][0].content

    # Cleanup
    await state_service.stop()
    await session_service.stop()


@pytest.mark.asyncio
async def test_langgraph_runner_error_handling():
    """Test LangGraph runner error handling."""

    class ErrorRunner(Runner):
        def __init__(self) -> None:
            super().__init__()
            self.framework_type = "langgraph"

        async def query_handler(
            self,
            msgs,
            request: AgentRequest = None,
            **kwargs,
        ):
            """Handler that raises an exception."""
            raise ValueError("Test error")

    # Setup services
    state_service = InMemoryStateService()
    session_service = InMemorySessionHistoryService()

    await state_service.start()
    await session_service.start()

    # Create runner
    runner = ErrorRunner()
    runner.state_service = state_service
    runner.session_service = session_service

    # Create request
    request = AgentRequest(
        input=[
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ],
        user_id="test_user",
        session_id="test_session",
    )

    # Run the agent and expect error to be handled gracefully
    with pytest.raises(ValueError, match="Test error"):
        async for response, is_last in runner.run(request):
            pass  # Should not reach here

    # Cleanup
    await state_service.stop()
    await session_service.stop()


if __name__ == "__main__":
    pytest.main([__file__])
