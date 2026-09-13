from dotenv import load_dotenv
# use groq ai because it is free
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
     model="openai/gpt-oss-120b"
)

result = model.invoke("Hello")

print(result.content)