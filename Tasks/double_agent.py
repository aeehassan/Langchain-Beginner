# An agent that can answer generic questions
# and solve math problems only. Its logic is
# to minimize api costs by selcting the model
# based on task complexity
#
# Bug:
# It works well for math questions. However,
# for generic questions, they're are handled
# poorly. It says there are no fns to handle
# such and then returns a tool call dict.
#

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.agents import create_agent

# Our models
model_one = ChatOllama(model="llama3.2:3b", temperature=0)
model_two = ChatOllama(model="mistral", temperature=0)


# determine prompt nature
def get_prompt_nature(prompt: str) -> str:
    """Figure out the type of the prompt

    Args:
        prompt (str): User's exact prompt

    Returns:
        str: Whether it's a Math or Generic question
    """
    model = ChatOllama(model="gemma3:1b", temperature=0)

    response = model.invoke(
        [
            SystemMessage("""You are to determine the nature of a user's
                         prompt. If it is a mathematics question, output 'math' else if
                         it is a Generic question, output 'gen'. Your output must 
                         be either one of them. It can't be anything 
                         beyond them."""),
            HumanMessage(prompt + " Is this math or gen"),
        ]
    )

    return response


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Select model based on prompt nature"""
    nature: str = get_prompt_nature(
        request.state["messages"][-1].content
    ).content.lower()

    print(nature)

    model = None
    if nature.__contains__("math"):
        model = model_two
    else:
        model = model_one

    # I think the problem has something to do with
    # here.
    return handler(request.override(model=model))


# Calculator tools
@tool
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


@tool
def sub(a: float, b: float) -> float:
    """Subtract two numbers"""
    return a - b


@tool
def mul(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


@tool
def div(a: float, b: float) -> float:
    """Divide two numbers"""
    if b == 0:
        return "Can't divide by 0"
    else:
        return a / b


agent_one = create_agent(
    model_one, tools=[add, mul, div, sub], middleware=[dynamic_model_selection]
)

state = agent_one.invoke({"messages": HumanMessage("Why is the sky blue")})
answer = state["messages"][-1].content
print(answer)
