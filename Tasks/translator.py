from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage, AIMessage

# Create the model
model = ChatOllama(model="gamma3:1b")

# Lay out instructions
print(
    """
        I am Corporate Translator. I translate informal to formal text.
        
        type 'exit' to terminate
    """
)

# Set the behaviour and memory of model
memory = [
    SystemMessage("""
                  You are an expert at translating any informal text to formal text.
                  You return a professional email style output that strictly follows
                  the following output. It is of the form:
                  
                  Dear Manager,
                  
                  [The formal text]
                  
                  Sincerely,
                  Abubakar
                  
                  There should be nothing that comes after Abubakar 
                  """),
    AIMessage("""
              So, to do this, should the salution and the sincerely part should be
              exactly as they are in the format -- from the words to how it is 
              positioned. No extra line spaces too?
              """),
    HumanMessage("Yess. Affirmative"),
    HumanMessage(""),
]

# Interaction bw Model and User
while True:
    prompt = input("User: ")

    if prompt == "exit":
        break

    # Add prompt to memory
    memory[-1] = HumanMessage(prompt)
    response = model.invoke(memory)

    print(f"Model: \n{response.content}")
