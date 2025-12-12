from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2:3b")

# Prompt Templates
## PromptTemplate
prompt = PromptTemplate.from_template(
    """You are a tutor for computer science students.
    You're to help undergrads have an easier learning experience of CS concepts.
    Explain the concept of {concept} in simple terms. The definition should not
    be more than 30 words. Include a brief practical example."""
)

c_one = prompt.format(concept="inheritance in object-oriented programming")

cs_response = model.invoke(c_one)
print("Prompt Template Response: \n")
print(f"{cs_response.content} \n")

# Chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a tutor for {major} students.
    You're to help undergrads have an easier""",
        ),
        (
            "human",
            """Explain the concept of {concept} in simple terms. 
        The definition should not be more than {word_limit} 
        words. Include a brief practical example.""",
        ),
    ]
)

c_two = prompt.invoke(
    {
        "major": "computer science",
        "concept": "encapsulation in object-oriented programming",
        "word_limit": 30,
    }
)

cs_response = model.invoke(c_two)

print("Chat Prompt Template Response: \n\n")
print("{cs_response.content} \n")
