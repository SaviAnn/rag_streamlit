import streamlit as st
import json
import requests
import os
import time
import torch
import pickle
from sentence_transformers import SentenceTransformer, util

st.title("⚡ Bible Wisdom Bot")

# 📌 **Show Loading Message**
loading_message = st.empty()
loading_message.info("⏳ Wait a minute! I am revising the Bible knowledge...")

# 📌 **Use GPU if Available**
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📌 **Load Model with GPU Support**
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return model

model = load_model()

# 📌 **Load JSON Data (Cached)**
file_path = "output.json"

@st.cache_data
def load_data():
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file).get("values", [])
            if not data:
                st.error("⚠️ Error: No data found in output.json!")
                st.stop()
        return data
    except json.JSONDecodeError:
        st.error("⚠️ Error: Invalid JSON format in output.json!")
        st.stop()

data = load_data()
chunk_texts = [item.get("content", "") for item in data]
chunk_ids = [item.get("chunk_id", "Unknown") for item in data]

# 📌 **Precompute & Save Embeddings to Avoid Recalculation**
embeddings_path = "embeddings.pkl"

@st.cache_resource
def compute_and_save_embeddings():
    if os.path.exists(embeddings_path):
        try:
            with open(embeddings_path, "rb") as f:
                embeddings = pickle.load(f)
        except Exception as e:
            st.error(f"⚠️ Error loading embeddings: {e}")
            st.stop()
    else:
        try:
            with torch.no_grad():
                embeddings = model.encode(
                    chunk_texts, 
                    convert_to_tensor=True, 
                    dtype=torch.float16,  
                    batch_size=128,
                    device=device
                )
            with open(embeddings_path, "wb") as f:
                pickle.dump(embeddings, f)
        except Exception as e:
            st.error(f"⚠️ Error generating embeddings: {e}")
            st.stop()
    return embeddings

embeddings = compute_and_save_embeddings()

# 📌 **Remove Loading Message After Completion**
loading_message.empty()

# 📌 **OpenAI API Details**
API_URL = "https://ragtest9908777201.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
API_KEY = os.getenv("OPENAI_API_KEY", "F0XafLT9zQUTFRCRg3HC3EB2DyIgeVgSDJo41cP1qS3aertfIM9dJQQJ99BBACHYHv6XJ3w3AAAAACOGysU2")

# 📌 **Retrieve Passage**
def retrieve_passage(query):
    try:
        with torch.no_grad():
            query_embedding = model.encode(query, convert_to_tensor=True, dtype=torch.float16, device=device)
            scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
            best_idx = scores.argmax().item()
        return chunk_texts[best_idx], chunk_ids[best_idx]
    except Exception as e:
        return "", "Unknown"

# 📌 **OpenAI API Call**
def chat_with_openai(prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        response_json = response.json()
        if "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        return "⚠️ Unexpected API response format!"
    except Exception as e:
        return f"⚠️ Error: {e}"

# 📌 **Chat History Management**
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the Bible..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    relevance_check = chat_with_openai(f"You are a Bible expert. Determine if this question is related to the Holy Bible: '{prompt}' If yes, respond with 'Relevant'. Otherwise, say 'Good question, but I am here to discuss the Holy Bible.'")
    
    if "Relevant" in relevance_check:
        relevant_passage, passage_id = retrieve_passage(prompt)
        response = chat_with_openai(f"You are a Bible expert. Answer this question: '{prompt}' using this passage: '{relevant_passage}'.")
        
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        if relevant_passage:
            with st.chat_message("assistant"):
                st.markdown(f"📖 **Reference Verse {passage_id}:** {relevant_passage}")
            st.session_state.messages.append({"role": "assistant", "content": f"📖 **Reference Verse:** {relevant_passage}"})
    else:
        with st.chat_message("assistant"):
            st.markdown("Good question, but I am here to discuss the Holy Bible.")
        st.session_state.messages.append({"role": "assistant", "content": "Good question, but I am here to discuss the Holy Bible."})