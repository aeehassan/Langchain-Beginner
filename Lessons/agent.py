from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    wrap_tool_call,
)
from langchain.tools import tool

# Static agent
model_one = ChatOllama(model="gemma3:1b", temperature=0)

agent_one = create_agent(model_one, tools=[])
response = agent_one.invoke(
    {
        "messages": [
            HumanMessage(
                "What is the capital of Nigeria? And describe Nigerians in one sentence"
            )
        ]
    }
)
print(response["messages"][1].content)

# Dynamic agent
model_two = ChatOllama(model="llama3.2:3b", temperature=0)


# Turns out handler is hidden in the langchain source code
# and automatically passed to the middleware function as the
# second argument
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Select model based on user input"""
    message = request.state["messages"][0].content

    if message.__contains__("nigerian"):
        model_two = ChatOllama(model="gemma3:1b")
    else:
        model_two = ChatOllama(model="llama3.2:3b")

    return handler(request.override(model=model_two))


agent_two = create_agent(model_two, tools=[], middleware=[dynamic_model_selection])
response = agent_two.invoke(
    {
        "messages": [
            HumanMessage("I am a nigerian. Tell me about my country in 20 words")
        ]
    }
)

print(response["messages"])


# Tool node
@tool
def search(query: str) -> str:
    """Search information on the web"""
    return f"Results for: {query}"


@tool
def download(files: str) -> str:
    """Download a file from the internet"""
    return f"Downloaded: {files}"


@tool
def upload(files: str) -> str:
    """Upload a file from the internet"""
    return f"Uploaded: {files}"


agent_three = create_agent(
    model_one,
    tools=[search, download, upload],
    middleware=[dynamic_model_selection],
)


# Tool error handling
@tool
def throw_value_error() -> bool:
    """Throw an exception on user request"""
    raise ValueError("Ikr :)")


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        msg = ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )
        print(msg)
        return msg


agent = create_agent(
    model=model_two, tools=[throw_value_error], middleware=[handle_tool_errors]
)
response = agent.invoke({"messages": [HumanMessage("Throw an exception error")]})
print(response)

for msg in response["messages"]:
    print(f"{msg.content} \n")
