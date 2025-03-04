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

# Retrieve up to 3 relevant passages with a similarity score threshold of 0.5
def retrieve_passage(query):
    try:
        with torch.no_grad():
            query_embedding = model.encode(query, convert_to_tensor=True, dtype=torch.float16, device=device)
            scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]  
            
            # Get top 3 matches
            top_k = min(3, len(scores))  
            top_indices = torch.topk(scores, top_k).indices.tolist()  
            
            # Filter by similarity score threshold of 0.5
            relevant_passages = [
                (chunk_texts[i], chunk_ids[i], scores[i].item()) 
                for i in top_indices if scores[i].item() >= 0.5
            ]
        
        return relevant_passages  # List of tuples: (text, id, score)
    
    except Exception as e:
        return []

# Function to interact with OpenAI API
def chat_with_openai(prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=6)
        response.raise_for_status()
        response_json = response.json()
        if "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        return "Unexpected API response format!"
    except Exception as e:
        return f"Error: {e}"

# Chat history management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "next_question" not in st.session_state:
    st.session_state.next_question = False
if "end_conversation" not in st.session_state:
    st.session_state.end_conversation = False  # Flag to end conversation

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Check if conversation has ended
if st.session_state.end_conversation:
    st.markdown("Thank you for using Bible Wisdom Bot!")
    st.stop()

# Display prompt for next question
if st.session_state.next_question:
    with st.chat_message("assistant"):
        st.markdown("How can I assist you further?")
    st.session_state.messages.append({"role": "assistant", "content": "How can I assist you further?"})
    st.session_state.next_question = False

# Handle user input
if prompt := st.chat_input("Ask about the Bible..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    relevance_check = chat_with_openai(
        f"You are a Bible expert. Determine if this question is related to the Holy Bible: '{prompt}' If yes, respond with 'Relevant'. Otherwise, say 'Good question, but I am here to discuss the Holy Bible.'"
    )
    
    if "Relevant" in relevance_check:
        relevant_passages = retrieve_passage(prompt)
        
        if relevant_passages:
            combined_passages = " ".join([p[0] for p in relevant_passages])
            response = chat_with_openai(f"""You are a Bible expert. Consider the following references as background knowledge and answer the original question: '{prompt}'. 
                                        The references are: '{combined_passages}'. Don't use other verses in your response and provide a clear and concise answer, don't mentiot that references were provided to you.""")
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            for passage, passage_id, score in relevant_passages:
                with st.chat_message("assistant"):
                    st.markdown(f"**Reference Verse {passage_id} (Score: {score:.2f}):** {passage}")
                st.session_state.messages.append({"role": "assistant", "content": f"**Reference Verse {passage_id} (Score: {score:.2f}):** {passage}"})
        else:
            with st.chat_message("assistant"):
                st.markdown("No relevant passages found (score < 0.5).")
            st.session_state.messages.append({"role": "assistant", "content": "No relevant passages found (score < 0.5)."})
