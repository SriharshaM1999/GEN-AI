from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional, Literal

class Review(BaseModel):
    key_themes: list[str] = Field(..., description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(..., description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(..., description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(None, description="Write the name of the reviewer")

load_dotenv()

model = ChatOpenAI()

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful                     
Review by Nitish Singh
""")

print(result.name)

# to convert pydantice model to a dictionary, we can use the .dict() method provided by pydantic.
result_dict = result.dict()
print(result_dict)

#to convert pydantic model to a json, we can use the .json() method provided by pydantic.
json_result = result.json() # or use result.modeel_dump_json() if you are using pydantic v2 
print(json_result)


"""
Notes:
------
1. In this code snippet, we have defined a Pydantic model called Review that represents the structure of the review 
data we want to extract from the input text. Each field in the model
------

2. Field: It is used to provide additional metadata and validation rules for the fields in the Pydantic model.
The description parameter is used to provide a description of the field, which can be helpful for documentation
we can also use other parameters like d
efault, 
regex, etc. to provide more specific validation rules for the fields.
lg, gt, le, ge, etc. to specify numerical constraints on the fields.
Example: 
cgpa: float = Field(description="The CGPA of the student", gt=0.0, le=10.0, default: 5.0)
"""