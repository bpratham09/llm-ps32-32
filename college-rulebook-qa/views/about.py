"""'How it works' page: architecture and design decisions, for viewers and evaluators."""
import streamlit as st

from rulebook_qa.config import ROOT_DIR, settings
from rulebook_qa.prompts import ANSWER_PROMPT
from rulebook_qa.ui import hero

hero("🧠 How it works",
     "Retrieval-Augmented Generation (RAG): find the right handbook passages first, "
     "then let the LLM answer only from them.")

st.markdown("#### Architecture")
st.image(str(ROOT_DIR / "assets" / "architecture.svg"), width="stretch")

st.markdown("#### Design decisions")
st.markdown(f"""
| Decision | Why |
|---|---|
| **RAG instead of fine-tuning** | Rules change every year; updating a PDF is easier than retraining a model |
| **Chunks of {settings.chunk_size} characters, {settings.chunk_overlap} overlap** | Small enough to retrieve one rule precisely, overlap stops rules being cut in half |
| **Header/footer removal** | The college name and page number repeat on every page and add noise to every chunk |
| **Page number stored on every chunk** | Lets every answer cite its page so students can verify it |
| **Temperature 0 + strict prompt** | Keeps answers factual and close to the handbook wording |
| **Fixed "not found" sentence** | The app can detect it reliably and avoids made-up answers |
| **Follow-up rewriting** | "What about practicals?" becomes a full question before searching |
| **FAISS cache keyed by file hash** | Each handbook is embedded only once, which saves time and API quota |
""")

st.markdown("#### Tech stack")
c = st.columns(3)
c[0].markdown("**Interface**  \nStreamlit (multi-page)")
c[1].markdown(f"**LLM**  \n`{settings.chat_model}` via {settings.llm_provider} API")
c[2].markdown(f"**Embeddings**  \n`{settings.embed_model}` ({settings.embed_provider}, local)" if settings.embed_provider == "fastembed" else f"**Embeddings**  \n`{settings.embed_model}`")
c = st.columns(3)
c[0].markdown("**Framework**  \nLangChain")
c[1].markdown("**Vector store**  \nFAISS")
c[2].markdown("**Testing**  \npytest + evaluation set")

with st.expander("📜 The answer prompt"):
    st.code(ANSWER_PROMPT, language="text")
