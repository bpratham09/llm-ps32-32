"""
College Rulebook Q&A - entry point.
Run:  streamlit run app.py
"""
import streamlit as st

from rulebook_qa.ui import inject_css, render_sidebar

st.set_page_config(page_title="College Rulebook Q&A", page_icon="📘", layout="wide")
inject_css()

pages = st.navigation([
    st.Page("views/chat.py", title="Ask the Rulebook", icon="💬", default=True),
    st.Page("views/evaluation.py", title="Evaluation", icon="📊"),
    st.Page("views/about.py", title="How it works", icon="🧠"),
])
render_sidebar()
pages.run()
