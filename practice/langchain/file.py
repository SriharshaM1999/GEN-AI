from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal, Optional
from langchain_openai import OpenAIEmbeddings;


load_dotenv()


class ExpectedFormat(BaseModel):
    topicName: str = Field('topic name')
    description: str = Field(description="Few lines about the topic")
    isTrending: Literal['True', 'False'] = Field(description='Is topic trending now')


pydanticOutputParser = PydanticOutputParser(pydantic_object=ExpectedFormat)

chatModel = ChatOpenAI(model_name="gpt-4o", temperature=0.7, max_tokens=1000)
embeddingModel = OpenAIEmbeddings();

promptTemplate1 = PromptTemplate.from_template(
    template = """
    HI..!, Describe about {topic}, follow {format_instructions}
""",
input_variable = ["topic"],
partial_variables={"format_instructions": pydanticOutputParser.get_format_instructions()},
validate_template = True,
)

promptTemplate2 = PromptTemplate.from_template(
    template = """
    HI..!, Describe about {topic}
""",
input_variable = ["topic"],
validate_template = True,
)

chain = promptTemplate1 | chatModel | pydanticOutputParser
reponse = chain.invoke({"topic": 'GEN AI'})


chain2 = promptTemplate2 | chatModel.with_structured_output(ExpectedFormat);
respose2= chain2.invoke({"topic": "ML"})

print("response ", reponse)
print("reponse2 ", respose2);



embeddingOfAString = embeddingModel.embed_query("this is a string");
print("embeddingOfAString " , embeddingOfAString);




#ChatPromptTemplate

chatPromptTemplate =ChatPromptTemplate(
            [
                ("system", "You are a helpful AI bot. Your name is {name}."),
                ("human", "Hello, how are you doing?"),
                ("ai", "I'm doing well, thanks!"),
                 MessagesPlaceholder(variable_name= "conversations"),
                ("human", "{user_input}"),
            ]
        )

conversations = [
    ("human", "what is API?"),
]


chain3 = chatPromptTemplate | chatModel

reponse4 = chain3.invoke({"name": 'BOB', "conversations":conversations , "user_input": "what are we discussing about"})
print(reponse4)

