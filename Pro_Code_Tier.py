import streamlit as st
import sys
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
from spellchecker import SpellChecker
import requests
import json
import streamlit_shadcn_ui as ui
import time

# Set page configuration
st.set_page_config(page_title="Navy Chat", page_icon="🚢", layout="centered")

# Set up SpellChecker
spell = SpellChecker()

# Function to interact with Ollama API
def query_ollama(prompt, model="llama3.2:latest"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None

# Sidebar
with st.sidebar:
    st.title("🚢 Medical Navy Chatbot 💬")
    st.write("Configure your chatbot settings below.")

# Connect to Zilliz Cloud cluster
CLUSTER_ENDPOINT = "https://in03-cf607103ea8262d.serverless.gcp-us-west1.cloud.zilliz.com"
TOKEN = "0a3ae0ae0608129e5e33848199bd46ea36c44d150106e446100a61d69b1813991815e17eb96eee35cba3ff49c759274aecc1e9ba"
connections.connect(uri=CLUSTER_ENDPOINT, token=TOKEN)

# Load model and tokenizer from Hugging Face Hub
tokenizer_hyb = AutoTokenizer.from_pretrained('BAAI/llm-embedder')
model = AutoModel.from_pretrained('BAAI/llm-embedder')
model.eval()
tokenizer_hyb.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer_hyb))

# Function to perform hybrid search
def hybrid_search(query_text):
    encoded_input = tokenizer_hyb([query_text], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
        query_embedding = torch.nn.functional.normalize(model_output[0][:, 0], p=2, dim=1)

    search_param_poster = {
        "data": query_embedding,
        "anns_field": "posterVector",
        "param": {"metric_type": "L2", "params": {"nprobe": 10}},
        "limit": 5
    }

    request_poster = AnnSearchRequest(**search_param_poster)
    rerank = WeightedRanker(1.0)
    collection_name = "Oct_31"
    collection = Collection(collection_name)
    collection.load()
    res = collection.hybrid_search(reqs=[request_poster], rerank=rerank, limit=5, output_fields=["metadata", "chunks"])

    summary_list = []
    for hit in res:
        for entity in hit:
            metadata = entity.entity.get("metadata")
            chunks = entity.entity.get("chunks")
            combined_string = f"{chunks}. This information is from {metadata.get('source')} on page {metadata.get('page')}."
            summary_list.append(combined_string)
    return summary_list

# Function to update query with spell correction
def updateQuery(query):
    return query

# Function to get answer from QA pipeline
def get_answer(question):
    search_results = hybrid_search(question)
    context = " ".join(search_results)
    prompt = f"  Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Try to utilize the input to answer this question, but dont mention to me if it is convoluded information and act as though it makes sense:\n\nQuestion:\n{question} \n\nContext:\n{context}  Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Try to utilize the input to answer this question, but dont mention to me if it is convoluded information and act as though it makes sense:\n\nQuestion:\n{question} "
    result = query_ollama(prompt)
    return result

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please enter your question."}]

# Main Streamlit app
st.title("💬 Chatbot")
st.caption("This chatbot is trained on the following PDFs shown below.")

ui.badges(badge_list=[("Humanitarian Assistance and Disaster Relief Aboard the USNS Mercy", "secondary"), 
                      ("US Navy Ship-Based Disaster Response", "secondary"), 
                      ("Sea Power: The US Navy and Foreign Policy", "secondary"), 
                      ("A Decade of Surgery Abroad the US Naval Ship Comfort", "secondary"), 
                      ("Hospital Ships Adrift?", "secondary")], 
                      class_name="flex gap-2", key="badges1")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input with spell correction and response display
user_input = st.chat_input("Enter your question:")
if user_input:
    corrected_input = updateQuery(user_input)
    st.session_state.messages.append({"role": "user", "content": corrected_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(corrected_input)
    
    # Placeholder for loading animation
    with st.chat_message("assistant"):
        placeholder = st.empty()  # Initialize a placeholder

        # Add loading message with fading animation (CSS)
        loading_html = """
        <style>
        .loading-message {
            animation: fadeInOut 3s ease-in-out infinite;
        }
        @keyframes fadeInOut {
            0% { opacity: 0; }
            50% { opacity: 1; }
            100% { opacity: 0; }
        }
        </style>
        <div class="loading-message">Generating response...</div>
        """
        placeholder.markdown(loading_html, unsafe_allow_html=True)

        # Simulate processing time (remove this in production)
        time.sleep(2)  # Simulating delay for fetching data

        # Get chatbot response
        answer = get_answer(corrected_input)
        result = ''
        for line in answer.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                token = data.get('response', '')
                result += token
                placeholder.write(result)

        # Add the completed answer to session state chat history
        st.session_state["messages"].append({"role": "assistant", "content": result})

