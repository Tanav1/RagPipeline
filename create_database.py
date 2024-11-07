from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json
from pathlib import Path
import os
import shutil
from dotenv import load_dotenv
import requests 
from huggingface_hub import configure_http_backend

file_path='../content/PDF-Extract-Kit/US_Comfort_nice.json'
chroma_path = "chroma"
data = json.loads(Path(file_path).read_text())

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'token'

def backend_factory() -> requests.Session:
    session = requests.Session()    
    session.verify = False    
    return session
configure_http_backend(backend_factory=backend_factory) 

model_name = "sentence-transformers/all-MiniLM-L6-v2"

requests.get("https://gateway.zscalergov.net:443/_sm_ctn")
hf = HuggingFaceEmbeddings(
    model_name=model_name,
)

def main():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_db(chunks)

def load_documents():
    loader = JSONLoader(
    # file_path='./content/PDF-Extract-Kit/US_Comfort_nice.json', #PDF Extract Kit JSON
    # jq_schema='.[] | .layout_dets[].text', #PDF Extract Kit JSON
    file_path='./PyMuPDF/PyMUPDF_output.json', #PyMuPDF JSON
    jq_schema='.[]', #PyMuPDF JSON
    text_content=False)
    data = loader.load()
    return data

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 500,
    length_function = len,
    add_start_index = True,)
    chunks = text_splitter.split_documents(documents)
    # print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # document = chunks[10]
    # pprint(document.page_content)
    # pprint(document.metadata)

    return chunks

def save_to_db(chunks):
    # clear db first
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
    db = Chroma.from_documents(
        chunks, HuggingFaceEmbeddings(), persist_directory=chroma_path
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {chroma_path}.")

if __name__ == "__main__":
    main()
