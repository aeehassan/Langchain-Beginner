# Dynamic prompt for an agent that helps with
# breakups.
#

from langchain_ollama import ChatOllama
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage

model = ChatOllama(model="llama3.2:3b")


class CustomState(AgentState):
    user_role: str


@dynamic_prompt
def prompt_via_role(request: ModelRequest) -> str:
    """Respond to the user based on their role"""

    role = request.state["user_role"]

    if role == "user":
        return """Assume the role of a nice and vibrant assistant.
                  Use emojis in your response. Your response should 
                  not be more than 50 words"""
    elif role == "admin":
        return """Assume the role of a strict and blunt assistant.
                  Use no emoji at all in your response. Your response should 
                  not be more than 50 words"""
    else:
        return """Just be nice. Your response should not be more than 
               50 words"""


agent = create_agent(
    model,
    middleware=[prompt_via_role],
    state_schema=CustomState,
)

state = {}

state = agent.invoke(
    {
        "messages": HumanMessage(
            "Hi!! My girlfriend broke up with me. Cheer me up please"
        ),
        "user_role": "user",
    }
)

print(state["messages"][-1].content)

state = {}

state = agent.invoke(
    {
        "messages": HumanMessage(
            "Hi!! My girlfriend broke up with me. Cheer me up please"
        ),
        "user_role": "admin",
    }
)

print(state["messages"][-1].content)
