# An error history tracker through its state
#

# Lessons
# typing is a class used to describe the
# nature of data that should be stored in a data structure
#

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import AgentState, create_agent
from typing import List, Dict
from random import random
from langgraph.types import Command
from langchain.messages import HumanMessage, ToolMessage

## Approach 1

model = ChatOllama(model="llama3.2:3b")


class CustomState(AgentState):
    errors: List[Dict[str, str]]


@tool
def get_response() -> str | Command:
    """Get a response from the model"""
    try:
        if random() <= 0.5:
            response = "Success!!"
            print(response)
            return response

        raise ValueError("Oops!! Something went wrong :(")

    except Exception as e:
        print(e)
        return e


agent = create_agent(model, tools=[get_response], state_schema=CustomState)

state = {}
errors = []

for i in range(0, 5):
    state = agent.invoke(
        {"messages": [HumanMessage("Give me a response")], "errors": errors}
    )
    toolmsg = state["messages"][-2]

    if toolmsg.content != "Success!!":
        errors.append({toolmsg.tool_call_id: toolmsg.content})


print(state["errors"])

## Approach 2 -- Wasn't a success but I learnt
# Using state_schema, if you wwant to access
# state within a tool, you can read and write.
# However, if you must read, you must assign a
# default value to it and for the write, you
# can add as many things as you want
#
# Also, reading from a tool is useless imo,
# however, writing to a state from it is so
# useful fr fr
#

# model = ChatOllama(model="llama3.2:3b")


# class CustomState(AgentState):
#     errors: List[Dict[str, str]]


# @tool
# def get_response(runtime) -> str | Command:
#     """Get a response from the model"""
#     try:
#         if random() <= 0.5:
#             print("Success!!")
#             return "Success!!"

#         raise ValueError("Oops!! Something went wrong :(")

#     # runtime.state.get("errors", [])

#     except Exception as e:
#         print(e)
#         return Command(
#             update={
#                 "errors": [{runtime.tool_call_id: str(e)}],
#                 "messages": [
#                     ToolMessage(
#                         content=str(e),
#                         tool_call_id=runtime.tool_call_id,
#                     )
#                 ],
#             }
#         )


# agent = create_agent(model, tools=[get_response], state_schema=CustomState)

# state = agent.invoke({"messages": [HumanMessage("Give me a response")], "errors": []})

# print(state["errors"])
