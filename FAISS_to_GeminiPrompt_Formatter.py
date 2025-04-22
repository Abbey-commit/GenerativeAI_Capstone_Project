
# 🔄 Convert FAISS Search Results into Gemini AI Studio Prompt Format

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Step 1: Load FAISS Index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.load_local("vectorstore/pubmed_faiss_index", embedding_model)

# Step 2: Define your query
query = "What are the emerging drug targets for psychosis?"

# Step 3: Retrieve top-k relevant document chunks
results = faiss_index.similarity_search(query, k=3)

# Step 4: Combine chunks into a formatted context block
context = "\n\n".join([doc.page_content for doc in results])

# Step 5: Format the Gemini AI Studio Prompt
prompt_template = f"""
You are a biomedical research assistant. Based on the following scientific literature:

{context}

Answer this question:
"{query}"
"""

# Step 6: Copy the output to AI Studio
print("✅ Paste this into Gemini AI Studio:")
print("="*80)
print(prompt_template)
print("="*80)
