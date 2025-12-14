from langchain_ollama import ChatOllama

# from langchain.tools import tool
import time

model = ChatOllama(model="gemma3:1b")

# Invocation
## Streaming invocation
response = model.stream("Define an adjective in 15 words.")

for chunk in response:
    print(chunk.content)
    print("---")

## These following two wont work coz response is an iterator obj.
## that can only run once
for chunk in response:
    print(chunk.content, end="")

for chunk in response:
    print(chunk.content, end="", flush=True)

## Batch invocation
responses = model.batch(
    [
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is quantum computing?",
    ]
)
for response in responses:
    print(response)


# Decorators in python
### Without args
## Tutorial example
def get_time(base):
    def enhancedfn():
        start = time.time()
        base()
        end = time.time()
        print(f"Task time: {end - start}s")

    return enhancedfn


@get_time
def make_cake():
    print("baking cake...")
    time.sleep(2)
    print("cake is done :)")


make_cake()


## My example
def count_to_ten(base):
    def enhancedfn():
        for i in range(1, 11):
            print(i)
        base()

    return enhancedfn


@count_to_ten
def make_cake():
    print("baking cake...")
    time.sleep(1)
    print("cake is done :)")


make_cake()


### With arguments
## Tutorial example
def get_time(base):
    def enhancedfn(*args, **kwargs):
        start = time.time()
        base(*args, **kwargs)
        end = time.time()
        print(f"Task time: {end - start}s")

    return enhancedfn


@get_time
def make_cake(cake_type, wait):
    print(f"We're baking {cake_type} cake")
    time.sleep(wait)
    print("We're done here")


make_cake("Vanila", 2)

## My example
memory = []


def add_to_history(base):
    def enhancedfn(*args, **kwargs):
        result = base(*args, **kwargs)
        memory.append(result)
        return result

    return enhancedfn


@add_to_history
def add(a, b):
    return a + b


@add_to_history
def times(a, b):
    return a * b


add(1, 3)
add(3, 4)
times(1, 3)
times(5, 3)
times(5, 2)
add(4, 1)
add(1, 9)
add(5, 3)

print(memory)

# Tool
# ...
