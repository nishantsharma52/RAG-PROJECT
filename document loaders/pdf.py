from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document loaders/Mern.pdf")

docs = data.load()

# splitter = TokenTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 10
#    )
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
   )

# print(docs[0])
chunks = splitter.split_documents(docs)

# print(len(chunks))

print(chunks[5].page_content)

