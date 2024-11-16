import streamlit as st
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import tempfile
from spellchecker import SpellChecker

spell = SpellChecker()

def updateQuery(query):
    corrected_words = []
    for word in query.split():
        corrected_word = spell.correction(word) if word in spell.unknown([word]) else word
        corrected_words.append(corrected_word)
    return ' '.join(corrected_words)

def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    return documents

def split_text_into_chunks(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_documents = text_splitter.split_documents(documents)
    return split_documents

def create_vector_store(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embedding_model)
    return db

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def get_answer(db, question):
    relevant_docs = db.similarity_search(question, k=5)
    context = " ".join([doc.page_content for doc in relevant_docs])  
    result = qa_pipeline({"question": question, "context": context})
    return result['answer']

def main():
    st.title("Hackathon RAG Model")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        documents = load_pdf(uploaded_file)
        split_documents = split_text_into_chunks(documents)
        db = create_vector_store(split_documents)
        
        user_input = st.text_input("Please enter your question:")
        if user_input:
            updated_q = updateQuery(user_input)
            if updated_q:
                answer = get_answer(db, updated_q)
                st.write(f"Answer: {answer}")
            else:
                st.write("Please enter a valid question.")
        else:
            st.write("Awaiting your question...")

if __name__ == "__main__":
    main()
