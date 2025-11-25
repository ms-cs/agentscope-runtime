# -*- coding: utf-8 -*-
import os
from contextlib import asynccontextmanager

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from my_tools import iqs_generic_search
from my_tools import read_plan_file
from my_tools import update_plan_file
from typing_extensions import Annotated
from typing_extensions import TypedDict

from agentscope_runtime.engine import AgentApp
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from agentscope_runtime.engine.services.agent_state import (
    InMemoryStateService,
)
from agentscope_runtime.engine.services.session_history import (
    InMemorySessionHistoryService,
)

# Set environment variables for LangChain
os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_OTEL_ONLY"] = "true"
# other environment variables, such as: OTEL_EXPORTER_OTLP_PROTOCOL, OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, OTEL_SERVICE_NAME

tools = [iqs_generic_search, update_plan_file, read_plan_file]
# Choose the LLM that will drive the agent
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt = """You are a proactive research assistant. """


# 定义 State 类型
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def print_user_question(state: AgentState) -> AgentState:
    """简单节点：打印用户问题"""
    user_input = state.get("messages", [])
    print(f"\n{'=' * 60}")
    print(f"📝 用户问题: {user_input}")
    print(f"{'=' * 60}\n")
    return state


def react_agent_node(state: AgentState) -> AgentState:
    """ReAct Agent 节点"""
    react_agent = create_agent(llm, tools, system_prompt=prompt)

    # 准备输入
    input_dict = {"messages": state["messages"]}

    print("\n🤖 Agent 开始处理...\n")

    # 流式调用
    for chunk in react_agent.stream(
        input_dict,
        stream_mode="updates",
    ):
        for step, data in chunk.items():
            print(f"step: {step}")
            if "messages" in data and data["messages"]:
                last_message = data["messages"][-1]
                if hasattr(last_message, "content_blocks"):
                    print(f"content: {last_message.content_blocks}")
                elif hasattr(last_message, "content"):
                    print(f"content: {last_message.content}")

    # 返回状态
    return {
        "messages": state["messages"],
    }


def build_graph():
    """构建 LangGraph 工作流"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("print_question", print_user_question)
    workflow.add_node("react_agent", react_agent_node)

    # 添加边
    workflow.add_edge(START, "print_question")
    workflow.add_edge("print_question", "react_agent")
    workflow.add_edge("react_agent", END)

    # 编译图
    graph = workflow.compile(name="langgraph_react_agent")
    return graph


# Create the AgentApp instance
agent_app = AgentApp(
    app_name="LangGraphAgent",
    app_description="A LangGraph-based research assistant",
)


# Initialize services as instance variables
@agent_app.init
async def init_func(self):
    self.state_service = InMemoryStateService()
    self.session_service = InMemorySessionHistoryService()

    await self.state_service.start()
    await self.session_service.start()


@agent_app.shutdown
async def shutdown_func(self):
    await self.state_service.stop()
    await self.session_service.stop()


@agent_app.query(framework="langgraph")
async def query_func(
    self,
    msgs,
    request: AgentRequest = None,
    **kwargs,
):
    # Extract session information
    session_id = request.session_id
    user_id = request.user_id

    # Load state if available
    state = await self.state_service.export_state(
        session_id=session_id,
        user_id=user_id,
    )

    # Build the graph agent
    search_agent = build_graph()

    # Process the messages through the agent with streaming
    # Using stream instead of invoke for better streaming support
    final_message = ""
    for chunk in search_agent.stream({"messages": msgs}):
        # Extract message content from chunk
        if isinstance(chunk, dict) and "messages" in chunk:
            messages = chunk["messages"]
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    content = last_message.content
                    if content:
                        final_message = content
                        # Yield each chunk for streaming
                        yield content, False  # (message, is_last=False for streaming)

    # Yield the final message with last flag set to True
    yield final_message, True

    # Save state if needed
    if final_message:
        state_data = {"last_message": final_message}
        await self.state_service.save_state(
            user_id=user_id,
            session_id=session_id,
            state=state_data,
        )


if __name__ == "__main__":
    agent_app.run(host="127.0.0.1", port=8090)
