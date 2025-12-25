# The task is just to create an agent with a
# custom state
#

from langchain_ollama import ChatOllama
from langchain.agents import AgentState, create_agent

model = ChatOllama(model="llama3.2:3b")


class CustomState(AgentState):
    visit_count: int


agent = create_agent(
    model,
    state_schema=CustomState,
    system_prompt="You are an agent who doesn't respond to any user prompts",
)

state = {}

for i in range(0, 5):
    state = agent.invoke({"visit_count": i + 1})

print(f"Final value: {state['visit_count']}")
