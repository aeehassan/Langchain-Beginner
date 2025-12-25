# Update state using Command
#
# Lessons learned
# - Langchain doesnt execute the way normal code runs
# - Your tool is part of your agent's runtime. So, it
# executes as though it weren't part of your code. So,
# global variables don't work.
#

from langchain_ollama import ChatOllama
from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from langgraph.types import Command
from langchain.messages import HumanMessage

model = ChatOllama(model="llama3.2:3b")


class CustomState(AgentState):
    counter: int


@tool
def update_counter() -> Command:
    """Increase counts"""

    return "Counter has been updated"


agent = create_agent(model, state_schema=CustomState)
state = {}

for i in range(0, 3):
    state = agent.invoke(
        {"messages": [HumanMessage("Increase count")], "counter": i + 1}
    )
    print(i)

print(state["counter"])
# print(state["messages"])
