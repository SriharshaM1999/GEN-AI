from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    

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

print(result['name'])


"""
Notes: 
------
Annotated: If you feel the LLM might not understand the context of a particular field of TypedDict, you can use Annotated to provide additional instructions or context to the model. 
This can help guide the model's understanding and improve the accuracy of the generated output.
Format : Annotated[Type, "Instruction or context for the model"]

-----
Optional[list[str]]: If you want to indicate that a particular field in the TypedDict is optional, you can use Optional from the typing module. 
This allows the model to understand that the field may not always be present in the output, and it can generate output accordingly.
-----

Literal: If you want to restrict the possible values for a particular field in the TypedDict, you can use Literal from the typing module.
-----

structured_model = model.with_structured_output(Review)
The with_structured_output method is used to create a new instance of the language model that is configured to produce output in a structured format defined by the Review TypedDict.
------

with structured output, it is not always guaranteed that model will always follow the types specified in the typedict, but it will try its best to adhere to the structure as much as possible.
But if you want the data validation, use pydantic instead of typedict.

"""

