from typing import Literal

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field


class Review(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the review"
    )


# Load environment variables
load_dotenv()

# Create model
model = ChatOpenAI()

# Parser
pydantic_parser = PydanticOutputParser(pydantic_object=Review)

# Sentiment classification prompt
sentiment_prompt = PromptTemplate(
    template=
"""
    Determine whether the following review is positive or negative.
    Review: {input} follow the {format_instructions}
""",
    input_variables=["input"],
    partial_variables={
        "format_instructions": pydantic_parser.get_format_instructions()
    },
)

# Branches
conditional_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        RunnableLambda(   # you can either use RunnableLamba or PromptTemplate here
            lambda _: (
                "The customer left a positive review. "
                "Thank them and ask for additional feedback."
            )
        )
        | model
        | StrOutputParser()
    ),
    (
        lambda x: x.sentiment == "negative",  # you can either use RunnableLamba or PromptTemplate here
        RunnableLambda(
            lambda _: (
                "The customer left a negative review. "
                "Apologize and ask for additional feedback."
            )
        )
        | model
        | StrOutputParser()
    ),
    RunnableLambda(lambda _: "Invalid sentiment"), # default case.
)

# Final chain
final_chain = (
    sentiment_prompt
    | model
    | pydantic_parser
    | conditional_chain # didnt append any parser, becuase, the branches already return string outputs
)

# Execute
response = final_chain.invoke(
    {"input": "The product is really good, I loved it!"}
)

print(response)

"""
Notes:
------
- The `RunnableBranch` takes in a list of branches, where each branch is a tuple of a condition and a  chain. 
The condition is a function that takes in the output of the previous step and returns a boolean. 
The runnable is executed if the condition is true.

----
- You can either use runnable lambda or a prompt template for the branches, depending on your use case. 
If you want to have a static response, you can use a runnable lambda. If you want to have a dynamic response based on the input,
 you can use a prompt template.

"""