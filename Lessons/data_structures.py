from pydantic import BaseModel, Field


# Student
class Student(BaseModel):
    """Biodata of a CS student"""

    matric_no: str = Field(description="Matric Number")
    name: str = Field(description="Full Name")
    department: str = Field(description="Department", min_length=4)
    cgpa: float = Field(ge=0.00, le=5.00, description="Current CGPA")


s1 = Student(
    matric_no="U22CS1060",
    name="Abubakar Abdulkadir Hassan",
    department="Computer Science",
    cgpa=4.97,
)
print(s1)


# Product details
class Product(BaseModel):
    """Details about a product for a store"""

    product_id: str = Field(description="Unique id", min_length=5, default="00000")
    name: str = Field(description="Product name", min_length=5)
    price: float = Field(description="Price of a unit", ge=0.00, le=100.00)
    description: str = Field(description="Product description", min_length=10)


p1 = Product(
    name="Getzener",
    price=99,
    description="""It comes in 
                    - Golden
                    - Brown
                    - Blue""",
)
print(p1)
