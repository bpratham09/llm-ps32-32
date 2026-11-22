"""Shared Streamlit helpers: styling, sidebar, and the cached QA engine."""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st

from . import __version__
from .config import has_api_key, key_env_name, settings
from .qa import RulebookQA
from .vectorstore import IndexInfo, build_or_load

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1100px;}
.hero {background: linear-gradient(135deg, #1E3A8A 0%, #2F5BEA 55%, #4F8DF7 100%);
       color: #fff; padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;}
.hero h1 {color: #fff; font-size: 1.9rem; margin: 0 0 .3rem 0; padding: 0;}
.hero p {color: #E3EBFF; margin: 0; font-size: 1.02rem;}
.chip {display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3);
       color: #fff; padding: .18rem .65rem; border-radius: 999px; font-size: .8rem; margin: .7rem .35rem 0 0;}
.cat-title {font-weight: 600; font-size: .95rem; margin: .2rem 0 .4rem 0; color: #1E3A8A;}
.src-card {border: 1px solid #E3E8F2; border-left: 4px solid #2F5BEA; border-radius: 10px;
           padding: .7rem .9rem; margin-bottom: .6rem; background: #FAFBFF;}
.src-card.dim {border-left-color: #B8C2D6;}
.badge {display: inline-block; background: #2F5BEA; color: #fff; font-size: .75rem; font-weight: 600;
        padding: .1rem .55rem; border-radius: 6px; margin-right: .4rem;}
.badge.grey {background: #8A94A8;}
.src-text {font-size: .86rem; color: #3B4458; margin-top: .4rem; white-space: pre-wrap;}
.muted {color: #6B7385; font-size: .82rem;}
.setup-card {border: 1px solid #E3E8F2; border-radius: 12px; padding: 1.1rem 1.3rem; background: #FAFBFF;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chip_html = "".join(f"<span class='chip'>{c}</span>" for c in (chips or []))
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p>{chip_html}</div>",
                unsafe_allow_html=True)


def file_digest(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def render_sidebar() -> None:
    """Sidebar shared by every page: API key, handbook choice and retrieval settings."""
    with st.sidebar:
        st.markdown("### 📘 Rulebook Q&A")
        st.caption(f"v{__version__} · Retrieval-Augmented Generation")

        if has_api_key():
            st.success("API key loaded", icon="🔑")
        else:
            key = st.text_input(f"API key ({key_env_name()})", type="password",
                                help="Or put it in the .env file so you don't have to paste it each time.")
            if key:
                os.environ[key_env_name()] = key.strip()
                st.rerun()

        st.markdown("**Handbook**")
        choice = st.radio("Handbook", ["Sample handbook", "Upload a PDF"], label_visibility="collapsed")
        if choice == "Sample handbook":
            st.session_state.pdf_path = str(settings.sample_pdf)
            st.session_state.pdf_name = settings.sample_pdf.name
        else:
            up = st.file_uploader("Upload your college handbook", type=["pdf"])
            if up:
                tmp = Path(tempfile.gettempdir()) / f"rulebook_{hashlib.md5(up.getvalue()).hexdigest()[:8]}.pdf"
                tmp.write_bytes(up.getvalue())
                st.session_state.pdf_path = str(tmp)
                st.session_state.pdf_name = up.name
            else:
                st.session_state.pdf_path = None

        with st.expander("⚙️ Retrieval settings"):
            st.session_state.top_k = st.slider("Chunks to retrieve (top-k)", 2, 8, settings.top_k)
            st.session_state.show_scores = st.toggle("Show similarity scores", value=False)
            st.caption(f"LLM: `{settings.chat_model}`  \nEmbeddings: `{settings.embed_model}` ({settings.embed_provider})  \n"
                       f"Chunk size {settings.chunk_size}, overlap {settings.chunk_overlap}")

        st.divider()
        st.caption("GenAI Mini Project · Roll No 32  \nCollege Rulebook Q&A")


@st.cache_resource(show_spinner=False)
def _load_engine(pdf_path: str, digest: str, top_k: int) -> tuple[RulebookQA, IndexInfo]:
    info = build_or_load(pdf_path)
    return RulebookQA(info.store, top_k=top_k), info


def get_engine() -> tuple[RulebookQA, IndexInfo] | None:
    """Return the QA engine for the selected handbook, or None (with a message) if not ready."""
    if not has_api_key():
        st.markdown(
            "<div class='setup-card'><b>👋 Welcome! One step to start:</b><br><br>"
            f"Paste your LLM API key in the sidebar, or set <code>{key_env_name()}</code> in the "
            "<code>.env</code> file (see <code>.env.example</code>).</div>",
            unsafe_allow_html=True)
        return None
    pdf_path = st.session_state.get("pdf_path")
    if not pdf_path:
        st.info("👈 Upload your college handbook PDF in the sidebar to begin.")
        return None

    digest = file_digest(Path(pdf_path))
    if st.session_state.get("active_digest") != digest:  # new handbook: start a fresh chat
        st.session_state.active_digest = digest
        st.session_state.messages = []
    try:
        with st.spinner("Reading and indexing the handbook (only the first time)..."):
            return _load_engine(pdf_path, digest, st.session_state.get("top_k", settings.top_k))
    except Exception as e:  # noqa: BLE001 - show any indexing problem to the user
        st.error(f"Could not index this PDF: {e}")
        return None
