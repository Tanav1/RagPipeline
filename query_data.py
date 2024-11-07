import argparse
# from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import retrieval_qa
from transformers import pipeline
from transformers import AutoModel
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

CHROMA_PATH = "/home/melody/repos/hackathon/pdf-extract-notebook/chroma"

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'


PROMPT_TEMPLATE = '''
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

'''

def main ():
    print("Starting...\n")
    #create CLI
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text

    query_text = "What issues do nurses face with Navy members at sea?"

    #prepare DB
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print("Embedded chunks...\n")

    #searches the DB bringing the top 3 chunks
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.2:
        print(f"Unable to find matching results.\nBest match was {results[0][1]}")
        return
    print("Searching DB...\n")
    
    # retrieve the context from the top 3 chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print("Retrieving top 3 chunks...\n")
    print(prompt)

    #trying mark's 
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    llm = HuggingFaceEmbeddings(pipeline=qa_pipeline)

    qa_chain = retrieval_qa.from_chain_type(llm=llm, retriever=db.as_retriever(), chain_type="stuff")

    result = qa_pipeline({
        "context": context_text,
        "question": query_text
    })

    print("\nAnswer: ", result)

    #initialize HF model for chat
    # model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", token="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")
    
    # model(PROMPT_TEMPLATE)
    # response = model(prompt, max_length=500, do_sample=True)
    # response_text = response[0]['generated_text']

    # sources = [doc.metadata.get("source", None) for doc, _score in results]
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)


if __name__ == "__main__":
    main()