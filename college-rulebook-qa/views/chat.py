"""Chat page: ask questions about the handbook and see answers with their source pages."""
import html
from datetime import datetime

import streamlit as st

from rulebook_qa.feedback import log_feedback
from rulebook_qa.ui import get_engine, hero

SUGGESTIONS = {
    "📅 Attendance": ["What is the minimum attendance to write exams?",
                     "My attendance is 68% due to illness. What can I do?"],
    "📝 Exams": ["Can I carry a smart watch into the exam hall?",
                "How do I apply for revaluation and what does it cost?"],
    "⚖️ Conduct": ["What happens if I use my phone in class?",
                   "How do I report ragging?"],
    "🏫 Campus": ["How many library books can I borrow?",
                 "What time must hostel students return?"],
}

hero("🎓 College Rulebook Q&A",
     "Ask anything about attendance, exams, conduct or campus rules. Every answer comes only from "
     "the official handbook and shows the page it came from.",
     ["Grounded answers", "Page citations", "Follow-up questions", "Says “not found” instead of guessing"])

engine = get_engine()
if engine is None:
    st.stop()
qa, info = engine
st.session_state.setdefault("messages", [])

# ---------- Handbook stats ----------
st.caption(f"📄 Answering from **{st.session_state.get('pdf_name', info.file_name)}**")
c1, c2, c3 = st.columns(3)
c1.metric("Pages indexed", info.n_pages)
c2.metric("Searchable chunks", info.n_chunks)
c3.metric("Chunks per answer (top-k)", qa.top_k)


# ---------- Helpers ----------
def ask(q: str) -> None:
    st.session_state.pending = q


def on_feedback(i: int) -> None:
    msg = st.session_state.messages[i]
    value = st.session_state.get(f"fb_{i}")
    if value is None:
        return
    question = st.session_state.messages[i - 1]["content"] if i else ""
    log_feedback(question, msg["content"], "up" if value == 1 else "down", msg.get("cited_pages", []))
    st.toast("Thanks for the feedback!", icon="🙏")


def render_sources(sources, show_scores: bool) -> None:
    cited = [s for s in sources if s.cited] or sources
    label = ", ".join(str(p) for p in sorted({s.page for s in cited}))
    with st.expander(f"📄 Sources · Page {label}"):
        for s in sources:
            badge = "badge" if s.cited else "badge grey"
            tag = "Cited" if s.cited else "Also retrieved"
            score = f" · distance {s.score:.3f}" if show_scores else ""
            st.markdown(
                f"<div class='src-card {'' if s.cited else 'dim'}'><span class='{badge}'>Page {s.page}</span>"
                f"<span class='muted'>{tag}{score}</span>"
                f"<div class='src-text'>{html.escape(s.text[:700])}{'…' if len(s.text) > 700 else ''}</div></div>",
                unsafe_allow_html=True)


def transcript() -> str:
    lines = [f"# College Rulebook Q&A - chat export ({datetime.now():%d %b %Y, %H:%M})", ""]
    for m in st.session_state.messages:
        who = "**Student:**" if m["role"] == "user" else "**Assistant:**"
        lines += [f"{who} {m['content']}", ""]
    return "\n".join(lines)


# ---------- Sidebar chat actions ----------
with st.sidebar:
    if st.session_state.messages:
        a, b = st.columns(2)
        if a.button("🧹 New chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()
        b.download_button("⬇️ Export", transcript(), file_name="rulebook_chat.md",
                          mime="text/markdown", width="stretch")

# ---------- Welcome / suggestions ----------
if not st.session_state.messages:
    st.markdown("#### Try asking")
    cols = st.columns(2, gap="large")
    for n, (cat, qs) in enumerate(SUGGESTIONS.items()):
        with cols[n % 2]:
            st.markdown(f"<div class='cat-title'>{cat}</div>", unsafe_allow_html=True)
            for j, q in enumerate(qs):
                st.button(q, key=f"sugg_{cat}_{j}", on_click=ask, args=(q,), width="stretch")

# ---------- Chat history ----------
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "📘"):
        if m["role"] == "user":
            st.markdown(m["content"])
            continue
        if m["found"]:
            st.markdown(m["content"])
            render_sources(m["sources"], st.session_state.get("show_scores", False))
        else:
            st.warning(m["content"], icon="🔎")
            st.caption("Tip: rephrase the question, or contact the college office for rules not in the handbook.")
        meta = f"⏱️ {m['latency']:.1f}s"
        if m["standalone"].strip().lower() != st.session_state.messages[i - 1]["content"].strip().lower():
            meta += f" · understood as: “{m['standalone']}”"
        st.caption(meta)
        st.feedback("thumbs", key=f"fb_{i}", on_change=on_feedback, args=(i,))

# ---------- New question ----------
question = st.chat_input("Ask about attendance, exams, conduct, fees...") or st.session_state.pop("pending", None)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📘"):
        with st.spinner("Searching the handbook..."):
            try:
                res = qa.ask(question, history=history)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "429" in msg or "quota" in msg.lower():
                    st.error("The API rate limit was hit. Please wait a minute and try again.")
                elif "404" in msg:
                    st.error("Model not found. Set a current CHAT_MODEL in your .env file.")
                else:
                    st.error(f"Something went wrong: {msg}")
                st.stop()
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": res.answer, "found": res.found, "sources": res.sources,
         "cited_pages": res.cited_pages, "standalone": res.standalone_question, "latency": res.latency_s},
    ]
    st.rerun()
