# An automated order validator -- Fabric business
#
# It takes customer texts and turns it into
# a validated order ticket using pydantic then
# returns it to the user as well.
# It enforces business logic on data before
# it ever hits the db.

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError
from langchain.messages import HumanMessage, SystemMessage

# Problem 1: Memory management by ollama
# Ollama creates a model instance for "model_with_structure",
# keeps it in memory and when "model_with_tools" is encountered,
# it tries creating another model instance which leads to a crash
# coz system Vram is small for such
#
# Fix 1:
# keep_alive=0: This forcibly makes the model unload from memory the
# moment a response is generated. However, it increases user wait
# time and leads to bad UX. And I also need to use the time fn to
# implement this wait
#
# Knowledge gains:
# - Context reduction: Context window is the amount of memory (VRam) that
# should be reserved for the model for processing a convo. While model
# weights are the exact size of the model. The strategy is to reduce
# that memory size to avoid crash. Parameter name is num_ctx.
# - It's not recommended to combine tool calling and structured output on
# the same model. It makes the llm confused
# - Temperature=0: This makes the model deterministic. It always returns
# the same output for the same input. And Temperature>0 makes the model
# non-deterministic - more creative and random in its responses.
#
# Fix 2:
# Use one model instance for both structured output and tool calling
# It leads to a more efficient resource management.
#

model = ChatOllama(model="llama3.2:3b", temperature=0, keep_alive=0)

messages = [
    SystemMessage("""You are an order validation assistant for a fabric business.
                     You will receive customer orders in text form.
                     Your task is to extract order details and return the validated
                     order details in a structured format."""),
    HumanMessage(""),
]


class OrderDetail(BaseModel):
    """Schema for storing order details
    extracted from customer order text"""

    customer_name: str = Field(
        description="Customer name",
    )
    fabric_type: str = Field(
        description="Fabric type e.g. cotton, silk, shadda, atampa",
    )
    color: str = Field(
        description="Fabric color e.g. red, blue, green",
    )
    yards: int = Field(description="Number of yards", ge=1, le=100)
    contact_number: str = Field(
        description="Customer contact number",
        pattern=r"^0\d{10}$",
    )


@tool(args_schema=OrderDetail)
def get_order(
    customer_name: str, fabric_type: str, color: str, yards: int, contact_number: str
) -> str:
    """Extract order details from customer order text"""
    order_detail = OrderDetail(
        customer_name=customer_name,
        fabric_type=fabric_type,
        color=color,
        yards=yards,
        contact_number=contact_number,
    )
    return order_detail.model_dump_json()


model_with_tool = model.bind_tools([get_order])

print("Type 'exit' to terminate the program")

while True:
    # Get user input
    user = input("Customer: ")
    if user == "exit":
        break

    messages[1] = HumanMessage(user)

    # Extract and validate order details
    get_order_request = model_with_tool.invoke(messages)

    # Return validated order details -- Either order or error message
    if get_order_request.tool_calls:
        get_order_request = get_order_request.tool_calls[0]

        try:
            response = get_order.invoke(get_order_request["args"])
            print(f"Agent: {response}")
        except ValidationError as e:
            errors = e.errors()
            fields = []

            # Get fields with errors
            for error in errors:
                field = error["loc"][0]
                fields.append(field)

            response = f"Your order has errors in the following fields: {', '.join(fields)}. Please correct them and try again."
            print(f"Agent: {response}")
    else:
        print("Agent: Could not process order. Please try again.")
