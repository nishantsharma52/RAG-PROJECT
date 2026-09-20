from dotenv import load_dotenv
# use groq ai because it is free
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

data = PyPDFLoader("document loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
)
chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [("system","you are AI that summerizes the text"),
     ("human","{data}")
     ]
)
model = ChatGroq(model="openai/gpt-oss-120b")

prompt = template.format_messages(data = docs[0].page_content)

result = model.invoke(prompt)

print(result.content)