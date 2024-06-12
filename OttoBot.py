#!/usr/bin/env python3
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import os
import argparse
import time
import requests

model = os.environ.get("MODEL", "Meemaw")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

def main():
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    llm = Ollama(model=model, callbacks=callbacks)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)
    while True:
        query = input("\nEnter your query\n ")
        if query.lower() == "exit":
            break
        if query.strip() == "":
            continue
        if query.lower() == "source":
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            continue

def parse_arguments():
    parser = argparse.ArgumentParser(description='MeemawGPT: Ask recipes to your documents without an internet connection, using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true', help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M", action='store_true', help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser.parse_args()

if __name__ == "__main__":
    main()