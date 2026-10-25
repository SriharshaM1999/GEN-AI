# LangChain: advanced and missing topics

These are topics from the playlist that your notes didn't cover, or only mentioned in passing. Read [01_langchain.md](01_langchain.md) first.

```python
from dotenv import load_dotenv; load_dotenv()
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
```

---

## 1. Structured output methods: `function_calling` vs `json_mode` vs `json_schema`
`with_structured_output(Schema, method=...)` can get structured data out of the model in three ways:
- **`function_calling`**: the schema is sent to the model as a tool, and the model "calls" it. Works on any model that supports tool calling. This is the safe default.
- **`json_mode`**: the model is only told to return *valid JSON*. **The schema is not sent**, so you have to describe the fields in the prompt yourself. Use it with models that have JSON mode but no tool calling.
- **`json_schema`** (OpenAI, newer models): the schema is sent and the provider **guarantees** the output matches it (strict mode). This is the most reliable.

Extras:
- `include_raw=True` returns `{"raw": AIMessage, "parsed": obj, "parsing_error": err}`. Useful for debugging and for seeing token usage.
- If the model supports none of these (small open-source models), go back to an **output parser**: `PydanticOutputParser` with format instructions in the prompt.

```python
class Movie(BaseModel):
    title: str
    year: int = Field(description="release year")

llm.with_structured_output(Movie).invoke("Best Nolan movie?")                        # function_calling
llm.with_structured_output(Movie, method="json_schema").invoke("Best Nolan movie?")  # strict
llm.with_structured_output(Movie, method="json_mode").invoke(
    "Best Nolan movie? Reply in JSON with keys 'title' and 'year'.")                   # describe the keys yourself

out = llm.with_structured_output(Movie, include_raw=True).invoke("Best Nolan movie?")
out["parsed"], out["raw"].usage_metadata
```

## 2. Local open-source models: `HuggingFacePipeline`
- `HuggingFaceEndpoint`: the model runs on HuggingFace's servers. Needs a token and the internet.
- `HuggingFacePipeline`: the model is **downloaded and runs on your machine**. No API cost, works offline, and the data stays private. It needs RAM or a GPU, and it's slow on a laptop, so stick to small models.
- Wrap either one in `ChatHuggingFace` to get the chat-model interface (messages in, `AIMessage` out).

```python
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

local_llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",   # about 2GB download on the first run
    task="text-generation",
    pipeline_kwargs={"temperature": 0.5, "max_new_tokens": 100},
)
model = ChatHuggingFace(llm=local_llm)
print(model.invoke("What is the capital of India?").content)
```

Bonus: `init_chat_model` switches provider by name, so the same code works for every model.
```python
from langchain.chat_models import init_chat_model
model = init_chat_model("openai:gpt-4o-mini", temperature=0)   # or "anthropic:...", "google_genai:..."
```

## 3. Why Runnables exist (the "toy LangChain" video)
- Early LangChain had a separate class for every chain (`LLMChain`, `SequentialChain`, `RetrievalQA`...), each with its own method names, and they couldn't be combined freely.
- The fix: give every component **one interface** (`invoke`). Anything that takes an input and returns an output can then be connected to anything else.
- The `|` operator works because `Runnable` defines `__or__`, which returns a sequence.

```python
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(self, x): ...
    def __or__(self, other):              # this is what makes a | b work
        return Sequence(self, other)

class Sequence(Runnable):
    def __init__(self, *steps): self.steps = steps
    def invoke(self, x):
        for step in self.steps:
            x = step.invoke(x)            # output of one step = input of the next
        return x

class Prompt(Runnable):
    def __init__(self, template): self.template = template
    def invoke(self, x): return self.template.format(**x)

class FakeLLM(Runnable):
    def invoke(self, prompt): return f"LLM answer to: {prompt}"

class Upper(Runnable):                    # stands in for a parser
    def invoke(self, text): return text.upper()

chain = Prompt("Tell me about {topic}") | FakeLLM() | Upper()
chain.invoke({"topic": "cricket"})
```

## 4. Choosing a PDF loader
- `PyPDFLoader`: plain text PDFs. Fast, one Document per page. Use this by default.
- `PyMuPDFLoader`: faster, and keeps more layout and metadata.
- `PDFPlumberLoader`: PDFs with **tables**.
- `UnstructuredPDFLoader`: **scanned** or image PDFs (uses OCR) and mixed layouts. Heavy to install.
- `AmazonTextractPDFLoader` / Azure Document Intelligence: managed OCR for production.
- Many PDFs in a folder: `DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)` + `lazy_load()`.

## 5. Custom document loader (`BaseLoader`)
Use this when no built-in loader fits your source (an internal API, a database, a special file format). Implement `lazy_load`. `load()` comes for free.

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class LineLoader(BaseLoader):
    """One Document per non-empty line of a text file."""
    def __init__(self, path: str):
        self.path = path

    def lazy_load(self):
        with open(self.path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    yield Document(page_content=line.strip(), metadata={"source": self.path, "line": i})

docs = LineLoader("cricket.txt").load()
```

---

## Self-test
1. With `method="json_mode"`, why must the prompt describe the fields?
2. When would you choose `HuggingFacePipeline` over `HuggingFaceEndpoint`?
3. Which Python method makes `prompt | llm` work?
4. Which loader would you use for a scanned PDF? And for a PDF full of tables?
5. Which method must a custom `BaseLoader` implement?
