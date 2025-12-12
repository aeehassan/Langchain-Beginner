from langchain_ollama import ChatOllama

# Create the model
model = ChatOllama(model="llama3.2:3b")

# Set the behavior and memory of the model
memory = [
    {
        "role": "system",
        "content": """
                    You are the dungeon guide in a Solo Leveling anime world
                    for any dungeon a hunter happens to enter. While the user
                    is the hunter. Your job is to describe the current scene
                    to our hunter. The hunter can only suggest an action he 
                    has decided to take to which you then narrate the 
                    consequences of such action
                   """,
    },
    {
        "role": "assistant",
        "content": """
                    Upon their entrance into the dungeon, I will only
                    tell them the rank, threats and treasures that lie 
                    in it.
                   """,
    },
    {
        "role": "human",
        "content": """
                    Ensure that your responses are brief, concise and straightforward.
                    They should be at most 25 words no matter what.
                    Also, remember that the ranks in order of priority are either S, A
                    , B, C, D or E.
                   """,
    },
]

# Instructions to the hunter
print(
    """
    You are now in the Solo Leveling world as a dungeon hunter.
    Interact with your dungeon guide to explore the dungeon.
    Type 'exit' to leave the dungeon.
    """
)

# Interaction
while True:
    prompt = input("User: ")

    if prompt == "exit":
        break

    # Add user's msg to memory
    memory.append({"role": "human", "content": prompt})

    response = model.invoke(memory).content

    # Add model's response to memory
    memory.append({"role": "assistant", "content": response})

    # Display to the hunter
    print(f"Dungeon guide: {response}")
