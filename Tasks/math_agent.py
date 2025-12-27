# This math agent is one that can only divide
# two whole numbers and return an integer
# output. It raises an error if any issue
# arises.
#

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage

# Model
model = ChatOllama(model="llama3.2:3b", temperature=0)

# Instructions to the user
print(
    """
    I am a math agent that can only perform the division
    operation perfectly using a divide tool.
    I have two rules:
    - I can only divide whole numbers
    - I cannot give a decimal result
    
    To quit, type 'exit'
    """
)


@tool
def safe_divide(num: int, den: int) -> int:
    """Divides two numbers"""
    if den == 0:
        raise ValueError(f"Cannot divide {num} by zero")

    result = num / den
    check = int(result)

    # The logic here is if result is acc
    # a float, float - int(float) is
    # always = 0.0
    if result - check != 0.0:
        raise ValueError("This agent only handles whole numbers")

    return int(result)


agent = create_agent(model, tools=[safe_divide])

while True:
    user = input("User: ")
    if user == "exit":
        break
    try:
        # Invoke returns a dict of agent's current state
        state = agent.invoke({"messages": [HumanMessage(user)]})
        print(f"Agent: {state['messages'][-1].content}")
    except ValueError as v:
        print(f"Agent: {v}")
