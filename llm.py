import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)