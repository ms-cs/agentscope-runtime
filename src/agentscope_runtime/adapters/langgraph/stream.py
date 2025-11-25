# -*- coding: utf-8 -*-
# pylint: disable=too-many-nested-blocks,too-many-branches,too-many-statements
"""Streaming adapter for LangGraph messages."""
import copy
import json

from typing import AsyncIterator, Tuple, List, Union, Dict, Any
from urllib.parse import urlparse

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolMessage,
    HumanMessage,
    SystemMessage,
)

from ...engine.schemas.agent_schemas import (
    Message,
    TextContent,
    ImageContent,
    AudioContent,
    DataContent,
    FunctionCall,
    FunctionCallOutput,
    MessageType,
)


def _update_obj_attrs(obj, **attrs):
    for key, value in attrs.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
    return obj


def _convert_dict_to_text_content(data: Dict[str, Any]) -> str:
    """Convert a dictionary to a text representation."""
    if not isinstance(data, dict):
        return str(data)

    # Handle the specific case: {'generate_joke': {'joke': 'This is a joke about ice cream and cats'}}
    # Convert to a more readable format
    lines = []
    for node_name, node_output in data.items():
        if isinstance(node_output, dict):
            lines.append(f"Node '{node_name}' output:")
            for key, value in node_output.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append(f"{node_name}: {node_output}")

    return "\n".join(lines)


async def adapt_langgraph_message_stream(
    source_stream: AsyncIterator[Tuple[Union[BaseMessage, Dict], bool]],
) -> AsyncIterator[Message]:
    """
    Optimized version of the stream adapter for LangGraph messages.
    Reduces code duplication and improves clarity.
    """
    last_msg_hash = None
    message = None

    async for msg, last in source_stream:
        msg = copy.deepcopy(msg)
        current_msg = msg
        if not message:
            message = Message(type=MessageType.MESSAGE, role="assistant")
            message = _update_obj_attrs(message, metadata={})
            yield message.in_progress()

        # Handle dict results from langgraph (e.g., {'generate_joke': {'joke': '...'}})
        if isinstance(current_msg, dict):
            # Create hash for message identification
            msg_content = json.dumps(current_msg, sort_keys=True)
            msg_hash = hash(msg_content)

            # Convert dict to text content
            text_content = _convert_dict_to_text_content(current_msg)

            # Create text content
            text_delta_content = TextContent(
                delta=True,
                index=None,
                text=text_content,
            )
            text_delta_content = message.add_delta_content(
                new_content=text_delta_content,
            )
            yield text_delta_content

            # Complete the message if this is the last chunk
            if last and message is not None:
                message = _update_obj_attrs(message, metadata={})
                yield message.completed()

            # Update last message hash
            last_msg_hash = msg_hash

        # Handle BaseMessage instances
        elif isinstance(current_msg, BaseMessage):
            # Create hash for message identification
            msg_content = str(current_msg.content) + str(
                getattr(current_msg, "tool_calls", [])
            )
            msg_hash = hash(msg_content)

            # Determine message role based on type
            if isinstance(current_msg, AIMessage):
                role = "assistant"
                # Handle tool calls in AIMessage
                if (
                    hasattr(current_msg, "tool_calls")
                    and current_msg.tool_calls
                ):
                    for tool_call in current_msg.tool_calls:
                        call_id = tool_call.get(
                            "id", f"tool_call_{hash(str(tool_call))}"
                        )
                        tool_name = tool_call.get("name", "unknown_tool")
                        tool_args = tool_call.get("args", {})

                        # Create plugin call message
                        plugin_call_message = Message(
                            type=MessageType.PLUGIN_CALL,
                            role=role,
                        )
                        plugin_call_message = _update_obj_attrs(
                            plugin_call_message,
                            metadata={},
                        )
                        yield plugin_call_message.in_progress()

                        # Create function call data
                        function_call_data = FunctionCall(
                            call_id=call_id,
                            name=tool_name,
                            arguments=json.dumps(
                                tool_args, ensure_ascii=False
                            ),
                        )

                        data_content = DataContent(
                            index=None,
                            data=function_call_data.model_dump(),
                            delta=True,
                        )
                        data_content = plugin_call_message.add_delta_content(
                            new_content=data_content,
                        )
                        yield data_content.completed()

                        plugin_call_message = _update_obj_attrs(
                            plugin_call_message,
                            metadata={},
                        )
                        yield plugin_call_message.completed()
                        continue

            elif isinstance(current_msg, ToolMessage):
                # Handle tool output messages
                call_id = current_msg.name.replace("tool_output_", "")
                plugin_output_message = Message(
                    type=MessageType.PLUGIN_CALL_OUTPUT,
                    role="tool",
                )

                # Create function call output data
                function_output_data = FunctionCallOutput(
                    call_id=call_id,
                    name="tool_output",
                    output=json.dumps(current_msg.content, ensure_ascii=False),
                )

                data_content = DataContent(
                    index=None,
                    data=function_output_data.model_dump(),
                )
                plugin_output_message.content = [data_content]
                plugin_output_message = _update_obj_attrs(
                    plugin_output_message,
                    metadata={},
                )
                yield plugin_output_message.completed()
                continue

            elif isinstance(current_msg, HumanMessage):
                role = "user"
            elif isinstance(current_msg, SystemMessage):
                role = "system"
            else:
                role = "assistant"  # Default fallback

            # Create message if not exists or if it's a new message
            if not message or msg_hash != last_msg_hash:
                message = Message(type=MessageType.MESSAGE, role=role)
                message = _update_obj_attrs(
                    message,
                    metadata={},
                )
                yield message.in_progress()

            # Handle content - could be string or list
            content = current_msg.content
            if isinstance(content, str):
                if content:  # Only process non-empty content
                    text_delta_content = TextContent(
                        delta=True,
                        index=None,
                        text=content,
                    )
                    text_delta_content = message.add_delta_content(
                        new_content=text_delta_content,
                    )
                    yield text_delta_content
            elif isinstance(content, list):
                # Handle list content (could be mixed text and other types)
                for item in content:
                    if isinstance(item, str):
                        if item:  # Only process non-empty content
                            text_delta_content = TextContent(
                                delta=True,
                                index=None,
                                text=item,
                            )
                            text_delta_content = message.add_delta_content(
                                new_content=text_delta_content,
                            )
                            yield text_delta_content
                    elif isinstance(item, dict):
                        # Handle dict content like images, etc.
                        item_type = item.get("type", "")
                        if item_type == "text":
                            text_content = item.get("text", "")
                            if text_content:
                                text_delta_content = TextContent(
                                    delta=True,
                                    index=None,
                                    text=text_content,
                                )
                                text_delta_content = message.add_delta_content(
                                    new_content=text_delta_content,
                                )
                                yield text_delta_content
                        # TODO: Handle other content types like images, etc.

            # Complete the message if this is the last chunk
            if last and message is not None:
                message = _update_obj_attrs(
                    message,
                    metadata={},
                )
                yield message.completed()

                # Reset message for next content
                message = None

            # Update last message hash
            last_msg_hash = msg_hash

        else:
            # Only use for last message, yield None,True
            if last and message is not None:
                message = _update_obj_attrs(message, metadata={})
                yield message.completed()
