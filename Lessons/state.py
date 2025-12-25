from langchain.agents import AgentState, create_agent
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2:3b")


class CustomState(AgentState):
    user_preferences: dict


agent = create_agent(model, tools=[], state_schema=CustomState)
# The agent can now track additional state beyond messages
response = agent.invoke(
    {
        "messages": [{"role": "user", "content": "I prefer technical explanations"}],
        "user_preferences": {"style": "technical", "verbosity": "detailed"},
    }
)

for msg in response["messages"]:
    print(f"{msg.content} \n")
