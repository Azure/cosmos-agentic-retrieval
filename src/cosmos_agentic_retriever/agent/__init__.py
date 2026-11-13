"""A search agent that answers questions with the MCP toolkit's tools.

The agent runs a short chat loop: it asks a language model which tool to call,
runs the tool, shows the model the results, and repeats until the model answers
or a turn cap is hit.
"""

from cosmos_agentic_retriever.agent.chat_client import ChatClient
from cosmos_agentic_retriever.agent.loop import (
    AgentResult,
    LlmTurn,
    run_agent_search,
)
from cosmos_agentic_retriever.agent.tool_contracts import (
    InvalidToolCall,
    Tool,
    ToolCall,
    ToolCallResponse,
    ToolExecutionResult,
    execute_tool_call,
)

__all__ = [
    "AgentResult",
    "ChatClient",
    "InvalidToolCall",
    "LlmTurn",
    "Tool",
    "ToolCall",
    "ToolCallResponse",
    "ToolExecutionResult",
    "execute_tool_call",
    "run_agent_search",
]
