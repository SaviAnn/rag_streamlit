import streamlit as st
import json
import requests
import os
import torch
import pickle
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
st.title("⚡ Bible Wisdom Bot")

# Use GPU if Available
device = "cuda" if torch.cuda.is_available() else "cpu"

# *Load Model with GPU Support
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return model

model = load_model()

# Load JSON Data (Cached)
file_path = "output.json"

@st.cache_data
def load_data():
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file).get("values", [])
    return data

data = load_data()
chunk_texts = [item.get("content", "") for item in data]
chunk_ids = [item.get("chunk_id", "Unknown") for item in data]

# Precompute & Save Embeddings to Avoid Recalculation
embeddings_path = "embeddings.pkl"

@st.cache_resource
def compute_and_save_embeddings():
    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            return pickle.load(f)
    
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
    
    return embeddings

embeddings = compute_and_save_embeddings()

# OpenAI API Details
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Retrieve up to 3 relevant passages with a similarity score threshold of 0.4
def retrieve_passage(query):
    with torch.no_grad():
        query_embedding = model.encode(query, convert_to_tensor=True, dtype=torch.float16, device=device)
        scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]  
        top_k = min(3, len(scores))  
        top_indices = torch.topk(scores, top_k).indices.tolist()  
        return [(chunk_texts[i], chunk_ids[i], scores[i].item()) for i in top_indices if scores[i].item() >= 0.4]


# Function to interact with OpenAI API

def chat_with_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-4",  
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    return response.choices[0].message.content

# Function to update prompt based on memory
def update_prompt(prompt):
    past_messages = [m["content"] for m in st.session_state.messages[-3:]]  # Last 3 messages
    print('History', past_messages)
    
    if not past_messages:
        return prompt  # No past context, return original prompt
    
    update_request = f"""Rephrase the given question to make it self-contained by incorporating necessary 
        context from the past messages: {past_messages}.
        Ensure the question remains very concise and does not introduce any new information. 

        Rules:
        1. If the question already makes sense without the context (i.e., it explicitly mentions the subject, place, or time), return it unchanged.
        2. If the question is ambiguous due to missing context, replace references like 'he/she/it/they' with the correct subject from past messages.
        3. Keep the wording natural and avoid unnecessary changes.
        4. All questions are about the Bible. No need to add it.

        Original prompt: '{prompt}'
        Return only the rephrased question, nothing else."""

    updated_prompt = chat_with_openai(update_request)
    
    return updated_prompt if updated_prompt else prompt
MAX_HISTORY = 20  

# Initialize memory in session state
if "messages" not in st.session_state:
    st.session_state.messages = []


if len(st.session_state.messages) > MAX_HISTORY:
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
# Initialize memory in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask about the Bible..."):
    st.chat_message("user").markdown(prompt)


    st.session_state.messages.append({"role": "user", "content": prompt})

   
    prompt = update_prompt(prompt)
    print('Updated Prompt:', prompt)

    relevance_check = chat_with_openai(
        f"You are a Bible expert. Determine if this question is related to the Holy Bible: '{prompt}'. If yes, respond with 'Relevant'. Otherwise, say 'Good question, but I am here to discuss the Holy Bible.'"
    )
    
    if "Relevant" in relevance_check:
        relevant_passages = retrieve_passage(prompt)
        
        if relevant_passages:
            combined_passages = " ".join([p[0] for p in relevant_passages])
            response = chat_with_openai(f"""You are a Bible expert. Consider the following references as background knowledge and answer the original question: '{prompt}'. 
                                         The references are: '{combined_passages}'. Don't use other verses in your response and provide a clear and concise answer, 
                                         don't mention that references were provided to you.""")

            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            references_list = [
                f"*📖 Reference Verse {passage_id} (Score: {score:.2f}):* {passage}"  
                for passage, passage_id, score in relevant_passages
            ]

            reference_message = "**Answer is based on these references:**  \n" + "  \n".join(references_list)
            with st.chat_message("assistant"):
                st.markdown(reference_message)
            st.session_state.messages.append({"role": "assistant", "content": reference_message})

        else:
            response = chat_with_openai(f"You are a Bible expert. Answer the original question: '{prompt}'. Provide a clear and comprehensive answer.")
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
    else:
        fallback_response = "Good question, but I am here to discuss the Holy Bible."
        with st.chat_message("assistant"):
            st.markdown(fallback_response)
        st.session_state.messages.append({"role": "assistant", "content": fallback_response})
