from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch, RunnableLambda


load_dotenv()

chatModel = ChatOpenAI();

RunnablePParallel


"""
Runnables: 

RunnableSerial, RunnableParalle, RunnableBranch, RunnableLamba, Runnable

"""




# Runnables, Docloader, Splitter, VectorStore, VectorDb, 