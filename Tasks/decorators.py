# Decorators in python
## Assignment 1
## Make a simple calculator with
## memory capability using decorators

# The memory of our calculator
memory = []


# The decorator that adds any operation
# performed to the memory
def add_to_history(base):
    def enhancedfn(*args):
        ops = {"add": "+", "div": "/", "times": "*", "sub": "-"}
        # Store the function obj's name
        name = base.__name__
        result = base(*args)

        # Extract the operator and operands
        operation = ops[name]
        operands = [*args]

        # Add the operation to the memory
        memory.append(f"{operands[0]} {operation} {operands[1]} = {result}")
        return result

    return enhancedfn


@add_to_history
def add(a, b):
    """Add two numbers"""
    return a + b


@add_to_history
def times(a, b):
    """Multiply two numbers"""
    return a * b


@add_to_history
def div(a, b):
    """Divide two numbers"""
    if b == 0:
        return None
    else:
        return a / b


@add_to_history
def sub(a, b):
    """Subtract two numbers"""
    return a - b


# Perform some operations
add(1, 3)
sub(3, 2)
times(5, 3)
div(60, 5)

# Display the memory state after operations
print(memory)
