from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from re import fullmatch
from langchain.messages import ToolMessage, HumanMessage
from langchain.agents import create_agent

model = ChatOllama(model="llama3.2:3b")


@tool
def validate_phone(contact: str) -> str:
    """ "Validate number as a Nigerian phone number"""
    if not fullmatch(r"^0\d{10}$", contact):
        raise ValueError(f"{contact} is not a valid Nigerian Phone Number")
    else:
        # return AIMessage("{contact} is valid")
        return f"{contact} is a valid Nigerian Phone Number"


@wrap_tool_call
def tool_error_handler(request, handler):
    """Handle error during tool execution"""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(content=f"{str(e)}", tool_call_id=request.tool_call["id"])


agent = create_agent(model, tools=[validate_phone], middleware=[tool_error_handler])

state = agent.invoke({"messages": [HumanMessage("My number is 12345")]})
messages = state["messages"]

for msg in messages:
    print(msg.content + "\n")
