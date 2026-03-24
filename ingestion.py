import os
import ssl
from config import API_KEY


os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

from langchain_groq import ChatGroq   # pyright: ignore[reportMissingImports]
from langchain_huggingface import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_chroma import Chroma  # pyright: ignore[reportMissingImports]

print("--- Step 1: Loading Vector Store ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="vector_db", embedding_function=embeddings)

print("--- Step 2: Initializing Groq LLM ---")
llm = ChatGroq(
    temperature=0, 
    groq_api_key=API_KEY, 
    model_name="llama-3.3-70b-versatile"
)

question = "What is the main goal of the linear regression model described in the document?"
print(f"\nQuestion: {question}")

print("--- Step 3: Searching PDF Context ---")
docs = vectorstore.similarity_search(question, k=3)
context = "\n---\n".join([doc.page_content for doc in docs])

prompt = f"Answer the question based ONLY on the following context:\n{context}\n\nQuestion: {question}"

print("--- Step 4: Generating Answer via Groq ---")
try:
    response = llm.invoke(prompt)
    print("\n" + "="*20)
    print("AI RESPONSE (Groq):")
    print(response.content)
    print("="*20)
except Exception as e:
    print(f"\nAn error occurred: {e}")