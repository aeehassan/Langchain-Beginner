# A customer support agent that changes
# its response based on the user status
#
from langchain_ollama import ChatOllama
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.agents import create_agent
from langchain.messages import HumanMessage

model_one = ChatOllama(model="llama3.2:3b", temperature=0)
model_two = ChatOllama(model="mistral", temperature=0.8)


@wrap_model_call
def dynamic_model_selector(request: ModelRequest, handler) -> ModelResponse:
    """Select model based on request urgency"""
    prompt = request.state["messages"][0].content

    if prompt.__contains__("URGENT"):
        model = model_two
        request.override(
            model=model,
            system_message="You are a Senior Manager. Be extremely apologetic and helpful.",
        )
    else:
        model = model_one
        request.override(
            model=model,
            system_message="You are a Junior Assistant. Be concise.",
        )

    return handler(request)


agent_one = create_agent(
    model_one,
    tools=[],
    middleware=[dynamic_model_selector],
)

state = agent_one.invoke(
    {"messages": [HumanMessage("URGENT: Where is the road to Amazon from Nigeria")]}
)
response = state["messages"][-1].content

print(response)
