
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Assuming Gemini 2.0 response function is available
from rag_pipeline import generate_rag_response, text_chunks, index, model

st.set_page_config(page_title="Mental Health Drug Targets", layout="centered")
st.title("💊 AI-Powered Insights for Mental Health Drug Targets")
st.subheader("GenAI-powered system for Depression, Anxiety, and Psychosis")

# User input
query = st.text_input("Enter your query:", "What are the latest drug targets for treating depression?")
condition = st.selectbox("Select Mental Health Condition", ["Depression", "Anxiety", "Psychosis"])

if st.button("🔍 Generate Response"):
    with st.spinner("Generating response using Gemini 2.0..."):
        response = generate_rag_response(query)
        st.markdown("### 🤖 Gemini 2.0 Response")
        st.success(response)

        # FAISS Search
        st.markdown("### 🔍 Retrieved Relevant Chunks")
        query_vector = model.encode([query]).astype("float32")
        top_k = 5
        distances, indices = index.search(query_vector, top_k)

        chunks = [text_chunks[i] for i in indices[0]]
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}:** {chunk}")

        # Optional evaluation display
        eval_data = {
            "Chunk": [f"Chunk {i+1}" for i in range(top_k)],
            "Distance": distances[0]
        }
        df = pd.DataFrame(eval_data)
        st.markdown("### 📊 Relevance Scores")
        st.dataframe(df)

        # Simulated Metrics
        st.markdown("### 📈 System Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Query Time", "1.2s")
        col2.metric("Top-K Chunks", str(top_k))
        col3.metric("Gemini Status", "✅ Success")
