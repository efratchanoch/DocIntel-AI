\# DocIntel AI: Retrieval-Augmented Generation (RAG) System



DocIntel AI is a technical implementation of a RAG pipeline designed for semantic information retrieval from PDF document repositories. The system integrates vector-based search with Large Language Models (LLMs) to provide context-aware responses based on private data sets.



\## Technical Architecture



The application is structured into a multi-stage pipeline:



1\. \*\*Document Ingestion\*\*: Utilizes recursive character splitting to discretize PDF text into manageable chunks.

2\. \*\*Vector Embedding\*\*: Generates 384-dimensional dense vector representations using the `all-MiniLM-L6-v2` transformer model.

3\. \*\*Persistent Indexing\*\*: Employs \*\*Chroma DB\*\* for high-dimensional vector storage and similarity search.

4\. \*\*Contextual Retrieval\*\*: Implements a Top-K retrieval strategy to fetch relevant document segments based on cosine similarity.

5\. \*\*Inference\*\*: Leverages the \*\*Groq\*\* inference engine (Llama-3.3-70B) for low-latency response generation.



\## Technology Stack



\* \*\*Core Framework\*\*: LangChain

\* \*\*User Interface\*\*: Streamlit

\* \*\*Inference Engine\*\*: Groq API (Llama-3.3-70B-Versatile)

\* \*\*Vector Database\*\*: Chroma DB

\* \*\*Embeddings\*\*: HuggingFace Transformers



\## System Features



\* \*\*Streaming Interface\*\*: Real-time token streaming for model responses.

\* \*\*Source Attribution\*\*: Metadata tracking to provide exact source snippets and page references for every response.

\* \*\*Session Management\*\*: Persistent chat history within the application state.

\* \*\*Environment Resilience\*\*: Integrated SSL and CA bundle configurations for restricted network environments.



\## Installation \& Setup



\### 1. Environment Setup

```bash

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate

pip install -r requirements.txt

````



\### 2\\. Configuration



Create a `config.py` file in the root directory and add your API key:



```python

\# config.py

API\_KEY = "your\_groq\_api\_key"

```



\*Note: You can obtain an API key from https://www.google.com/search?q=console.groq.com/keys.\*



\### 3\\. Execution



To index documents and initialize the vector store:



```bash

python ingest.py

```



To launch the interactive chat interface:



```bash

streamlit run app.py

```



\## Project Status



The current implementation covers the full RAG cycle, including document processing, vector indexing, and the interactive chat interface. Future iterations will focus on multi-document support and advanced re-ranking strategies.

```

