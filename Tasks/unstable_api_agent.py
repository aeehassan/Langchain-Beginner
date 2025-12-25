from langchain_ollama import ChatOllama
from langchain.tools import tool
from random import random
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, HumanMessage
from langchain.agents import create_agent

model = ChatOllama(model="llama3.2:3b")


@tool
def api_service() -> str:
    """Runs for any prompt"""
    if random() < 0.5:
        raise RuntimeError("Service temporarily unavailable")
    return "success"


@wrap_tool_call
def tool_error_handler(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(content=f"{str(e)}", tool_call_id=request.tool_call["id"])


agent = create_agent(model, tools=[api_service], middleware=[tool_error_handler])

state = agent.invoke({"messages": [HumanMessage("Call the unstable service")]})
messages = state["messages"]

for msg in messages:
    print(msg.content + "\n")
