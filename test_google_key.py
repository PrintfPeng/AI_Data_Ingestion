from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print("API_KEY:", api_key[:8], "...")  # แค่ดูว่ามันอ่านได้จริง

emb = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=api_key,
)

vec = emb.embed_query("hello world")
print("Embedding len:", len(vec))
print("First 5 numbers:", vec[:5])
