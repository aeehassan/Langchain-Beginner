from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, HumanMessage
from langchain.agents import create_agent

model = ChatOllama(model="llama3.2:3b")


@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    if b == 0:
        raise ValueError(f"Can't divide {a} by 0")

    return a / b


@wrap_tool_call
def tool_error_handler(request, handler):
    """Handles tool execution exception"""
    try:
        return handler(request)
    except Exception as e:
        error = ToolMessage(
            content=f"An error occured during tool execution: {str(e)}",
            tool_call_id=request.tool_call["id"],
        )
        return error


agent = create_agent(
    model,
    tools=[divide],
    middleware=[tool_error_handler],
)

state = agent.invoke({"messages": HumanMessage("Divide 5 by 0")})
messages = state["messages"]

for msg in messages:
    print(msg.content + "\n")
