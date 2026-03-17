import os
import ssl

# SSL / Rimon filtering workaround (keep this at the very top)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st  # pyright: ignore[reportMissingImports]

from config import API_KEY

from langchain_chroma import Chroma  # pyright: ignore[reportMissingImports]
from langchain_groq import ChatGroq  # pyright: ignore[reportMissingImports]
from langchain_huggingface import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]


st.set_page_config(page_title="DocIntel AI", page_icon="📄", layout="centered")


def inject_theme() -> None:
    st.markdown(
        """
<style>
/* --- App background + typography --- */
.stApp {
  background: radial-gradient(1200px 600px at 10% 5%, #f7f9fc 0%, #f4f6fb 35%, #f2f5fa 100%);
  color: #0f172a;
}
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}

/* --- Header sizing --- */
h1, h2, h3 {
  letter-spacing: -0.02em;
}

/* --- Sidebar polish --- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
}
section[data-testid="stSidebar"] * {
  color: #e2e8f0 !important;
}
.di-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.06);
}
.di-brand .logo {
  width: 38px;
  height: 38px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.95) 0%, rgba(99, 102, 241, 0.95) 100%);
  box-shadow: 0 10px 25px rgba(37, 99, 235, 0.22);
  font-size: 18px;
}
.di-brand .title {
  font-weight: 700;
  font-size: 16px;
  line-height: 1.15;
}
.di-brand .subtitle {
  font-size: 12px;
  opacity: 0.85;
}
.di-badge {
  display: inline-block;
  padding: 6px 10px;
  margin-top: 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  background: rgba(34, 197, 94, 0.16);
  border: 1px solid rgba(34, 197, 94, 0.28);
}

/* --- Chat messages (subtle borders, soft cards) --- */
div[data-testid="stChatMessage"] {
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 14px;
  padding: 10px 12px;
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(8px);
  box-shadow: 0 10px 24px rgba(2, 8, 23, 0.04);
}
div[data-testid="stChatMessage"] p {
  font-size: 15px;
  line-height: 1.55;
}

/* --- Chat input emphasis --- */
div[data-testid="stChatInput"] {
  border: 1px solid rgba(148, 163, 184, 0.32);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.85);
  box-shadow: 0 12px 30px rgba(2, 8, 23, 0.06);
}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="vector_db", embedding_function=embeddings)


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGroq:
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=API_KEY,
        temperature=0.0,
        streaming=True,
    )


def build_prompt(question: str, context: str) -> str:
    return f"""You are DocIntel AI, a helpful assistant for question answering over documents.

Answer the question based ONLY on the provided context.
If the answer is not contained in the context, say: "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""
def render_sidebar(vectorstore: Chroma) -> None:
    # Sidebar Branding with Dark Background (instead of white)
    st.sidebar.markdown(
        """
        <div class="di-brand" style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);">
          <div class="logo" style="color: white; font-weight: bold;">DI</div>
          <div>
            <div class="title" style="color: white;">DocIntel AI</div>
            <div class="subtitle" style="color: rgba(255, 255, 255, 0.6);">RAG Engine</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Chat Management
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.toast("Chat history cleared", icon="🧹")
        st.rerun()

    st.sidebar.divider()
    
    # System Info
    st.sidebar.markdown("### ⚙️ System")
    st.sidebar.markdown(f"""
    - **Model:** `Llama-3.3-70B`
    - **DB:** `Chroma`
    """)

    st.sidebar.divider()

    # Metrics & Health
    st.sidebar.markdown("### 🛠️ Index Health")
    db_exists = os.path.exists("vector_db")
    
    # Secure chunk counting
    try:
        doc_count = vectorstore._collection.count() if vectorstore else 0
    except Exception:
        doc_count = "Error"

    st.sidebar.write(f"📂 **Storage:** `{'Ready' if db_exists else 'Missing'}`")
    st.sidebar.write(f"🧱 **Chunks:** `{doc_count}`")

    st.sidebar.divider()
    st.sidebar.caption("v1.0 | Developed by Efrat Chanoch")

def ensure_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def stream_answer(llm: ChatGroq, prompt: str):
    for chunk in llm.stream(prompt):
        text = getattr(chunk, "content", None)
        if text:
            yield text


def sources_expander(docs) -> None:
    with st.expander("View Sources"):
        if not docs:
            st.write("No sources were retrieved.")
            return

        for i, doc in enumerate(docs, start=1):
            meta = getattr(doc, "metadata", {}) or {}
            page = meta.get("page")
            source = meta.get("source") or meta.get("file_path") or meta.get("filename")

            header_bits = [f"Snippet {i}"]
            if source:
                header_bits.append(str(source))
            if page is not None:
                header_bits.append(f"page {page}")

            st.markdown(f"**{' · '.join(header_bits)}**")
            st.code((doc.page_content or "").strip())


def welcome_screen() -> None:
    st.markdown(
        """
<div style="padding: 14px 14px 6px 14px; border: 1px solid rgba(148,163,184,.25); border-radius: 16px; background: rgba(255,255,255,.72);">
  <div style="font-size: 18px; font-weight: 750; letter-spacing: -0.02em;">Welcome to DocIntel AI</div>
  <div style="margin-top: 6px; color: rgba(15,23,42,.78);">
    Ask questions about your indexed documents. DocIntel AI will retrieve relevant context and answer with sources.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("**Try an example:**")
    examples = [
        "Summarize the main goal of the linear regression model in the document.",
        "List the key assumptions mentioned for linear regression.",
        "What are the evaluation metrics discussed, and what do they mean?",
    ]

    cols = st.columns(3)
    for idx, q in enumerate(examples):
        if cols[idx].button(q, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()


def main() -> None:
    inject_theme()
    st.title("DocIntel AI")
    st.caption("A retrieval-augmented chat interface powered by Groq + Chroma.")

    vectorstore = get_vectorstore()
    llm = get_llm()
    render_sidebar(vectorstore)

    ensure_chat_state()

    if not st.session_state.messages and not st.session_state.pending_question:
        welcome_screen()

    render_chat_history()

    user_question = st.chat_input(placeholder="Ask me about the documentation...")
    if not user_question and st.session_state.pending_question:
        user_question = st.session_state.pending_question
        st.session_state.pending_question = None

    if not user_question:
        return

    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        st.toast("Searching your vector store…", icon="🔎")
        with st.spinner("Retrieving relevant context…"):
            docs = vectorstore.similarity_search(user_question, k=3)

        context = "\n---\n".join(doc.page_content for doc in docs)
        prompt = build_prompt(user_question, context)

        placeholder = st.empty()
        full_text = ""

        for token in stream_answer(llm, prompt):
            full_text += token
            placeholder.markdown(full_text)

        sources_expander(docs)

    st.session_state.messages.append({"role": "assistant", "content": full_text.strip()})


if __name__ == "__main__":
    main()

