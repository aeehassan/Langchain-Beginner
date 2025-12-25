# An agent that can keep track of tool calls
# via a state
#

from langchain_ollama import ChatOllama
from langgraph.types import Command
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage

model = ChatOllama(model="llama3.2:3b")
num = 0


# You can't initialize a state in the schema defxn
# They must be initialized when summoned like in ToolRuntime()
# invoke() which is compulsory
#
class CustomState(AgentState):
    tool_call_count: int


@tool
def call_a_tool(runtime: ToolRuntime, call_count: int) -> Command:
    """Call a tool

    Args:
        runtime (ToolRuntime): The tool's runtime information
        call_count (int): The previous value of how many times the tool has been called
    """
    # Read current state
    current_count = runtime.state.get("tool_call_count", call_count)

    # Update state
    current_count += 1

    print(f"Tool called {current_count} times")
    return Command(
        update={
            "tool_call_count": current_count,
            "messages": [
                ToolMessage(
                    content=f"Success! The tool has been called {current_count} times.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


agent = create_agent(
    model,
    tools=[call_a_tool],
    state_schema=CustomState,
    system_prompt="Call the tool for every user message. Dont respond at all. There's a print statement to handle response",
)

for i in range(0, 5):
    state = agent.invoke(
        {
            "messages": [
                HumanMessage("Call a tool"),
            ],
            "tool_call_count": i,
        }
    )

print(f"Final count: {state['tool_call_count']}")
