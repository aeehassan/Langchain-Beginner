from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from IPython.display import display, Markdown

# Initialiizing a model
model = ChatOllama(model="gemma3:1b")

# Invocation: Single message
response = model.invoke("Explain quantum mechanics in one sentence.")
print(response.content)

# Messages
## Object notation
messages = [
    SystemMessage(
        content="""You are a tutor for computer science students. 
        You're to help undergrads have an easier learning experience of CS concepts."""
    ),
    HumanMessage(content="Explain polymorphism in object-oriented programming."),
]

cs_response = model.invoke(messages)
print(cs_response.content)

## Dictionary notation
messages = [
    {
        "role": "system",
        "content": """
                You are a tutor for computer science students.
                You're to help undergrads have an easier 
                learning experience of CS concepts. You do so
                by first defining the concept simply then you
                follow it up with a practical example that is 
                also brief and concise. You do not make use 
                of any analogies
                """,
    },
    {"role": "user", "content": "Explain polymorphism in OOP"},
]

cs_response = model.invoke(messages)

display(Markdown(cs_response.content))
