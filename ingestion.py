import os
from langchain_community.document_loaders import PyPDFLoader  # pyright: ignore[reportMissingImports]
from langchain_text_splitters import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]
from langchain_openai import OpenAIEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import Chroma  # pyright: ignore[reportMissingImports]
from config import API_KEY

# Load the PDF
pdf_path = "regression.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"Number of pages found: {len(pages)}")

# Split the loaded documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)
print(f"Total number of chunks created: {len(chunks)}")

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

# Create and persist vector store
print("Creating vector store... this might take a few seconds.")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db"
)


vectorstore.persist()

print("Vector store created and saved successfully!")