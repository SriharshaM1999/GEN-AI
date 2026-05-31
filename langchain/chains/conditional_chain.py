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
        ...,
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
    template="""
Determine whether the following review is positive or negative.

Review:
{input}

{format_instructions}
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
        RunnableLambda(
            lambda _: (
                "The customer left a positive review. "
                "Thank them and ask for additional feedback."
            )
        )
        | model
        | StrOutputParser()
    ),
    (
        lambda x: x.sentiment == "negative",
        RunnableLambda(
            lambda _: (
                "The customer left a negative review. "
                "Apologize and ask for additional feedback."
            )
        )
        | model
        | StrOutputParser()
    ),
    RunnableLambda(lambda _: "Invalid sentiment"),
)

# Final chain
final_chain = (
    sentiment_prompt
    | model
    | pydantic_parser
    | conditional_chain
)

# Execute
response = final_chain.invoke(
    {"input": "The product is really good, I loved it!"}
)

print(response)