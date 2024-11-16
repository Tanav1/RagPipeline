import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import streamlit_shadcn_ui as ui

# Set up the Streamlit app configuration
st.set_page_config(
    page_title="Navy Chat",
    page_icon="🚢",
    layout="centered"
)

# Configure Google Gemini AI
genai.configure(api_key="YOUR_API_KEY")
generation_config = {
    "temperature": 0.2,
    "max_output_tokens": 8192,
    "top_k": 64,
    "top_p": 0.8,
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Function to generate response using Google Gemini model
def get_ai_response(question):
    try:
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        return "Sorry, I couldn't process that request."

# Sidebar configuration
with st.sidebar:
    st.title("🚢 Medical Navy Chatbot 💬")
    st.write("Configure your chatbot settings below.")
    st.write("Adjust model parameters:")
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    max_tokens = st.slider("Max Output Tokens", 500, 8192, 8192)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please enter your question."}]

# Main Streamlit app interface
st.title("💬 Chatbot")
st.caption("This chatbot is trained on the following PDFs shown below.")

ui.badges(badge_list=[("Humanitarian Assistance and Disaster Relief Aboard the USNS Mercy", "secondary"), 
                      ("US Navy Ship-Based Disaster Response", "secondary"), 
                      ("Sea Power: The US Navy and Foreign Policy", "secondary"), 
                      ("A Decade of Surgery Abroad the US Naval Ship Comfort", "secondary"), 
                      ("Hospital Ships Adrift?", "secondary")], 
                      class_name="flex gap-2", key="badges1")

# User input
user_question = st.chat_input("Enter your question:")
if user_question:
    st.session_state["messages"].append({"role": "user", "content": user_question})

    # Generate response
    with st.spinner("Generating response..."):
        answer = get_ai_response(user_question)
    st.session_state["messages"].append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
