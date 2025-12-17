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
from langchain_core.prompts import PromptTemplate


model = ChatOllama(model="llama3.2:3b", temperature=1)

messages = [
    SystemMessage("""You are an order validation assistant for a fabric business.
                     You will receive customer orders in text form.
                     Your task is to extract order details and return the validated
                     order details in a structured format."""),
    HumanMessage(""),
]


class OrderDetail(BaseModel):
    """Schema for order details"""

    customer_name: str = Field(description="Name of the customer")
    fabric_type: str = Field(description="Type of fabric ordered")
    color: str = Field(description="Color of the fabric")
    quantity: int = Field(description="Quantity of fabric in meters", ge=1)
    contact_number: str = Field(
        description="Customer contact number", pattern=r"^0[0-9]{10}$"
    )


model_with_structure = model.with_structured_output(OrderDetail)


# Extract and validate order details
@tool
def get_order(message: str) -> str:
    """Extract order details from customer order text"""
    order_detail = None
    try:
        order_detail = model_with_structure.invoke(message)
        return order_detail
    except ValidationError as e:
        errors = e.errors()
        fields = []

        # Get fields with errors
        for error in errors:
            field = error["loc"][0]
            fields.append(field)

        # Add those errors in memory
        messages.append(
            PromptTemplate(
                template=f"""Tell the user what mistakes they made in their order
                         request in the following fields {fields}"""
            )
        )

        # Have the model return the error the user made
        error_msg = model.invoke(messages)
        return error_msg.content


print("Type 'exit' to terminate the program")

while True:
    # Get user input
    user = input("Customer: ")
    if user == "exit":
        break
    messages[1] = HumanMessage(user)

    # Return validated order details
    model_with_tool = model.bind_tools([get_order])
    get_order_request = model_with_tool.invoke(messages).tool_calls[0]

    response = get_order.invoke(get_order_request)
    print(f"Agent: {response.content}")
