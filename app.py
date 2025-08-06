import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Paths
MODEL_DIR = "./neo_outputs"
INDEX_DIR = "./vector_index"

# Load components
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForCausalLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

@st.cache_resource
def load_index():
    index = faiss.read_index(f"{INDEX_DIR}/faiss_index.idx")
    with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

model, tokenizer = load_model()
index, chunks = load_index()
embedder = load_embedder()
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔍 Retrieval Function
def retrieve_chunks(query, k=3):
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector), k)
    return [chunks[i] for i in indices[0]]

# 🤖 Generation Function
def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    attention_mask = (inputs != tokenizer.pad_token_id).long()
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=150,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

# 🧠 Streamlit UI
st.title("🧠 Medical Q&A Assistant (LLM + RAG)")
st.markdown("Ask a medical question. The answer is based on your custom-trained model.")

query = st.text_input("Enter your medical question:")

if query:
    with st.spinner("Retrieving context..."):
        top_chunks = retrieve_chunks(query)
    with st.spinner("Generating answer..."):
        answer = generate_answer(query, top_chunks)

    st.subheader("Answer:")
    st.write(answer)

    with st.expander("🔍 Retrieved Chunks"):
        for i, chunk in enumerate(top_chunks):
            st.markdown(f"**Chunk {i+1}:** {chunk}")
