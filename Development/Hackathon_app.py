"""

rerun if fails first time

"""
#import random
#import numpy as np
#import os
### FieldSchema, CollectionSchema, DataType,
from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
#import random
#import time
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import pipeline
import streamlit as st
from spellchecker import SpellChecker


spell = SpellChecker()

# Token
# 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

# Connect to your Zilliz Cloud cluster
CLUSTER_ENDPOINT = "https://in03-cf607103ea8262d.serverless.gcp-us-west1.cloud.zilliz.com"
TOKEN = "0a3ae0ae0608129e5e33848199bd46ea36c44d150106e446100a61d69b1813991815e17eb96eee35cba3ff49c759274aecc1e9ba"

# Step 1: Connect to the cluster
connections.connect(uri=CLUSTER_ENDPOINT, token=TOKEN)


 # Load model and tokenizer from Hugging Face Hub
tokenizer_hyb = AutoTokenizer.from_pretrained('BAAI/llm-embedder')
model = AutoModel.from_pretrained('BAAI/llm-embedder')
model.eval()
# Add a new padding token
tokenizer_hyb.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer_hyb))  # Resize the embeddings to accommodate the new token


# Step 6: Perform hybrid search based on user input
def hybrid_search( query_text):
    print('starting search')
    # Tokenize sentences
    encoded_input = tokenizer_hyb([query_text], padding=True, truncation=True, return_tensors='pt')

    # Compute embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Use CLS token embedding as sentence embedding
        query_embedding = model_output[0][:, 0]
        # Normalize embeddings for cosine similarity
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

    # Define the search parameters
    search_param_poster = {
        "data": query_embedding,  # The query embedding
        "anns_field": "posterVector",  # Field to search in (posterVector)
        "param": {
            "metric_type": "L2",  # L2 distance for similarity
            "params": {"nprobe": 10}  # Search efficiency
        },
        "limit": 5  # Return top 5 results
    }

    request_poster = AnnSearchRequest(**search_param_poster)
    # Combine the requests
    reqs = [request_poster]

    # Step 9: Choose a reranking strategy (e.g., weighted ranker)
    rerank = WeightedRanker(1.0)

    # Load the collection before searching
    collection_name = "Oct_31"
    collection = Collection(collection_name)
    collection.load()

    # Step 10: Execute hybrid search with output fields
    res = collection.hybrid_search(
        reqs=reqs,  # Multiple search requests
        rerank=rerank,  # Reranking strategy
        limit=5,  # Limit for final results
        output_fields=["metadata", "chunks"]  # Include metadata in the search result
    )
    # Summarize each result
    summary_list = []

    # Process and print the results
    for hit in res:
        # Each hit contains attributes like id, distance, and entity
        for entity in hit:
            metadata = entity.entity.get("metadata")
            chunks = entity.entity.get("chunks")
            # Combine chunk and metadata into one string
            combined_string = (
                f"{chunks}. This information is from {metadata.get('source')} on page {metadata.get('page')}."
            )
            summary_list.append(combined_string)
    print('ending search')
    return summary_list



def updateQuery(query):
    corrected_words = []
    for word in query.split():
        corrected_word = spell.correction(word) if word in spell.unknown([word]) else word
        corrected_words.append(corrected_word)
    return ' '.join(corrected_words)


qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", max_length=2000, max_answer_len=200, device = 0)


def get_answer(question):
    print('start get answer')
    prompt = f"Answer the question with detailed support based on the context. be as detailed as possible and provide the longest answer you can Question: {question}"

    search_results = hybrid_search(question)
    context = " ".join([result for result in search_results])  
    print("Context:", context)

    # Use the prompt as part of the question in the pipeline
    result = qa_pipeline(question=prompt, context=context)
    print("QA Result:", result)
    answer = result.get('answer', "No relevant answer found.") 
    print('end get answer')
    return answer




def main():
    st.title("Hackathon RAG Model")
    user_input = st.text_input("Please enter your question:")
    updated_q = updateQuery(user_input)
    if user_input:
        updated_q = updateQuery(user_input)
        if updated_q:
            answer = get_answer(updated_q)
            st.write(f"Answer: {answer}")
        else:
            st.write("Please enter a valid question.")
    else:
        st.write("Awaiting your question...")

if __name__ == "__main__":
    main()

