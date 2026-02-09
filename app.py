import inspect
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import streamlit as st

from rag import chat as chat_module
from rag import ingest as ingest_module
from rag.config import load_config


SUPPORTED_UPLOAD_TYPES = ("pdf", "txt", "md")


def resolve_callable(module: Any, names: Iterable[str]) -> Callable[..., Any] | None:
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


def call_with_supported_args(fn: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(fn)
    supported_args = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return fn(**supported_args)


def ensure_upload_dir() -> Path:
    if "upload_dir" not in st.session_state:
        st.session_state.upload_dir = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
    return st.session_state.upload_dir


def persist_uploads(files: list[st.runtime.uploaded_file_manager.UploadedFile]) -> list[Path]:
    upload_dir = ensure_upload_dir()
    saved_paths: list[Path] = []
    for uploaded in files:
        if uploaded is None:
            continue
        destination = upload_dir / f"{uploaded.name}"
        destination.write_bytes(uploaded.getbuffer())
        saved_paths.append(destination)
    return saved_paths


def normalize_citations(raw_citations: Any) -> list[str]:
    if raw_citations is None:
        return []
    if isinstance(raw_citations, list):
        return [str(item) for item in raw_citations]
    if isinstance(raw_citations, tuple):
        return [str(item) for item in raw_citations]
    return [str(raw_citations)]


st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("RAG Chatbot")

config = load_config()

with st.sidebar:
    st.header("Configuration")
    config_data = asdict(config)
    for key, value in config_data.items():
        display_value = "✅ Configured" if value else "⚠️ Missing"
        st.caption(f"{key}: {display_value}")
    st.divider()
    st.header("Index Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=list(SUPPORTED_UPLOAD_TYPES),
        accept_multiple_files=True,
    )
    index_button = st.button("Index")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []
if "index_status" not in st.session_state:
    st.session_state.index_status = "No index created yet."

if index_button:
    ingest_fn = resolve_callable(
        ingest_module, ["ingest_documents", "ingest_files", "ingest"]
    )
    if ingest_fn is None:
        st.session_state.index_status = (
            "Ingestion function not available yet. "
            "Add ingest_documents/ingest_files in rag/ingest.py."
        )
    elif not uploaded_files:
        st.session_state.index_status = "Please upload at least one file."
    else:
        saved_paths = persist_uploads(uploaded_files)
        try:
            response = call_with_supported_args(
                ingest_fn,
                files=[str(path) for path in saved_paths],
                config=config,
            )
            st.session_state.index_status = (
                response if isinstance(response, str) else "Indexing complete."
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state.index_status = f"Indexing failed: {exc}"

status_col, results_col = st.columns([2, 1])

with status_col:
    st.subheader("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask a question about your documents")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        chat_fn = resolve_callable(
            chat_module,
            ["chat", "ask", "answer_question", "query"],
        )
        if chat_fn is None:
            assistant_message = (
                "Chat function not available yet. "
                "Add chat/ask/answer_question in rag/chat.py."
            )
            citations = []
        else:
            try:
                response = call_with_supported_args(
                    chat_fn,
                    question=user_prompt,
                    messages=st.session_state.messages,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001
                response = f"Chat failed: {exc}"
            if isinstance(response, dict):
                assistant_message = response.get("answer", "")
                citations = normalize_citations(response.get("citations"))
            elif isinstance(response, tuple) and len(response) == 2:
                assistant_message, citations = response
                citations = normalize_citations(citations)
            else:
                assistant_message = str(response)
                citations = []

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_message}
        )
        st.session_state.last_citations = citations
        with st.chat_message("assistant"):
            st.markdown(assistant_message)

with results_col:
    st.subheader("Results")
    st.caption(st.session_state.index_status)
    if st.session_state.last_citations:
        st.markdown("**Citations**")
        for citation in st.session_state.last_citations:
            st.markdown(f"- {citation}")
    else:
        st.caption("No citations yet.")
