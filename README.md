# RagPipeline
Team: Tanav Thanjavuru (coach), Melody Llinas, Bruke Amare, Udom An, Miya Reese, & Mark Zhang

This project is for the 2024 UNH ML-Hackathon.

# Navy Medical Chatbot Project
-------------------------------


This repository contains the files for the "Navy Ship" chatbot, an interactive Streamlit-based application designed to answer questions based on a collection of Navy-related documents. It uses Llama 3.2 for semantic search, language processing, and spell correction to provide relevant responses.

### Project Files
- Pro_Code_Tier.py: The main script to launch the Navy Medical Chatbot in Streamlit.
- setingup_VEC.ipynb: Jupyter Notebook for data preparation, which loads and preprocesses PDF files, splits them into meaningful chunks, and prepares them for embedding and semantic search. (DO NOT NEED TO RUN, here for sharing implementation purposes)

### Prerequisites
Ensure you have the required libraries installed by running:

```pip install streamlit torch transformers pymilvus spellchecker requests streamlit-shadcn-ui PyPDF2 langchain sentence-transformers```
or 
```pip install -r requirements.txt```

## Running the Chatbot

### 1. Set up ollama in one terminal
Note: You MUST do this before running the streamlit app.

Run the following commands in your terminal:

```brew install ollama```
```ollama serve```
```ollama pull llama3.2```

Now leave this terminal running, and open another terminal.

### 2. Running the Streamlit App
Start Streamlit App: To launch the chatbot, run the following command:
```streamlit run Pro_Code_Tier.py```
