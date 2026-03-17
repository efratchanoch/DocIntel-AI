import os
import ssl
from config import API_KEY

# SSL filtering workaround
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

# Imports
from langchain_groq import ChatGroq  # pyright: ignore[reportMissingImports]
from langchain_huggingface import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_chroma import Chroma  # pyright: ignore[reportMissingImports]

# Load the vector store
print("Loading vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="vector_db", embedding_function=embeddings)

# Initialize the Groq chat model
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=API_KEY,
    temperature=0.0,
)

# Retrieval step
question = "What is the main goal of the linear regression model described in the document?"
print(f"\nQuestion: {question}")
print("Searching for relevant context...")

# Retrieve the 3 most relevant chunks
docs = vectorstore.similarity_search(question, k=3)
context = "\n---\n".join([doc.page_content for doc in docs])

# Augment + generate
prompt = f"""
Answer the question below based ONLY on the provided context.
If the information is not in the context, say you don't know.

Context:
{context}

Question: 
{question}

Answer:"""

print("Generating answer via Groq...")
try:
    response = llm.invoke(prompt)
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(response.content.strip())
    print("=" * 60)
except Exception as e:
    print(f"\nAn error occurred while generating the answer: {e}")