# RagPipeline
Team: Tanav Thanjavuru (coach), Melody Llinas, Bruke Amare, Udom An, Miya Reese, & Mark Zhang

This project is for the 2024 UNH ML-Hackathon.

Navy Medical Chatbot Project
-------------------------------


This repository contains the files for the "Navy Chat" chatbot, an interactive Streamlit-based application designed to answer questions based on a collection of Navy-related documents. It uses a combination of machine learning models for semantic search, language processing, and spell correction to provide relevant responses.

Project Files
Pro_Code_Tier.py: The main script to launch the Navy Medical Chatbot in Streamlit.
setingup_VEC.ipynb: Jupyter Notebook for data preparation, which loads and preprocesses PDF files, splits them into meaningful chunks, and prepares them for embedding and semantic search. (DO NOT NEED TO RUN, here for sharing implementation purposes)

Prerequisites
Ensure you have the required libraries installed by running:


# --- pip install streamlit torch transformers pymilvus spellchecker requests streamlit-shadcn-ui PyPDF2 langchain sentence-transformers
or 

# --- pip install -r requirements.txt

How to set up ollama 3.2. Must run before.

# --- ollama serve

# --- ollama pull llama3.2

How to run:

Start Streamlit App: To launch the chatbot, run the following command:

# --- streamlit run Pro_Code_Tier.py