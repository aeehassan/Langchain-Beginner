from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage, SystemMessage, AIMessage

# A model with tools to assist a student
# with studying for their courses with
# ease.

model = ChatOllama(model="llama3.2:3b")


## Simple schema defxn
# Tool(s) definition
@tool()
def explain_concept(concept: str) -> str:
    """Explain a given concept in simple terms.

    Args:
        concept: The concept that is to be defined
    """
    response = model.invoke(f"""Explain the concept of {concept} in simple terms.
                                It should not be more than 30 words. Also, there 
                                should be no examples nor analogies""")
    return response.content


# Tool integration
model_with_tools = model.bind_tools([explain_concept])

# Tool request
response = model_with_tools.invoke("Explain OOP in simple terms")
tool_call = response.tool_calls
print(tool_call)
# Tool call
response = explain_concept.invoke(tool_call[0])
response.content
response.text


## Advanced schema definition
# A tool that predicts the class of degree
# a student is likely to graduate with on
# a 5.0 scale


class Student(BaseModel):
    """Basic student information"""

    matric_no: str = Field(description="Matric number", min_length=9)
    department: str = Field(description="Department name in full")
    cgpa: float = Field(description="Cummulative Grade Point Average out of 5.00")


@tool(args_schema=Student)
def get_degree_class(matric_no: str, department: str, cgpa: float) -> str:
    """Get the class degree of a student"""
    message = [
        SystemMessage("""You are a degree class predictor for uni students.
                         You must chose one of the below options:
                         First class: It ranges from 4.50 to 5.00
                         Second class upper: It ranges from 3.50 to 4.49
                         Second class lower: It ranges from 2.40 to 3.49
                         Third class: It ranges from 1.50 to 2.39
                         Pass: It ranges from 1.00 to 1.49
                         Fail: Anything below 1.00 is a Fail"""),
        AIMessage("""I will one return the degree class - nothing more.
                     If the CGPA is 3.5 for instance, I will only return
                     'Second class upper'. So, that means my response 
                     will be at mst three words long"""),
        HumanMessage(f"""What is the class of degree of a student with a 
                     CGPA of {cgpa} in a 5.0 scale based university? """),
    ]
    degree = model.invoke(message)
    return f"Matric number: {matric_no} \nCGPA: {cgpa} \nDepartment: {department} \n\nDegree: {degree.content}"


model_with_tools = model.bind_tools([explain_concept, get_degree_class])
prompt = model_with_tools.invoke("""My matric number is U22CS1001. I am
                                    in CS department. My CGPA is 4.49""")
tool_call = prompt.tool_calls[0]
# tool_call
response = get_degree_class.invoke(tool_call)
print(response.content)
