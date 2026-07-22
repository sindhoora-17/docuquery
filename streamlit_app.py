"""
Streamlit UI for DocuQuery. Run: streamlit run streamlit_app.py
Shows the answer plus the source pages it came from — the trust feature.
"""
import os

from dotenv import load_dotenv
import streamlit as st

from generation import GeminiBackend
from rag import RAGPipeline

load_dotenv()

st.set_page_config(page_title="DocuQuery AI", page_icon="📄")
st.title("📄 DocuQuery AI")
st.caption("Ask questions about your PDFs. Answers cite the pages they came from.")


@st.cache_resource
def get_pipeline():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        st.error("GEMINI_API_KEY not set."); st.stop()
    return RAGPipeline(GeminiBackend(key))


pipeline = get_pipeline()
question = st.text_input("Your question")

if question:
    with st.spinner("Retrieving and answering..."):
        result = pipeline.answer(question)
    st.markdown("### Answer")
    st.write(result.text)
    if result.sources:
        st.markdown("### Sources")
        for s in result.sources:
            st.markdown(f"- {s}")
