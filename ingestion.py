import os
import ssl

# SSL workaround for filtered internet 
# os.environ['CURL_CA_BUNDLE'] = ''
# os.environ['PYTHONHTTPSVERIFY'] = '0'
# ssl._create_default_https_context = ssl._create_unverified_context

from langchain_community.document_loaders import PyPDFLoader  # pyright: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
from langchain_huggingface import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import Chroma  # pyright: ignore[reportMissingImports]

# Load the PDF document
pdf_path = "regression.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"Number of pages found: {len(pages)}")

# Split the document into chunks
# Strategy: 600 characters per chunk with 100 characters overlap to keep context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)
print(f"Total number of chunks created: {len(chunks)}")

# Initialize local Embeddings (Free & Offline-capable)
# This model runs on your CPU and doesn't require an API Key or Quota
print("Initializing local embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and persist the vector store using ChromaDB
print("Creating vector store... this might take a minute on the first run.")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db"
)

# Ensure data is saved to the disk
vectorstore.persist()

print("Vector store created and saved successfully in 'vector_db' folder!")