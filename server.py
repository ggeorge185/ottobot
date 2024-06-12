from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import requests
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama

model = os.environ.get("MODEL", "Mistral")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

class QueryRequest(BaseModel):
    query: str
    preferences: str

app = FastAPI()

# Adding CORS middleware if you are accessing the API from a frontend hosted on a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = Ollama(model=model)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa

qa_chain = get_qa_chain()

@app.post("/query")
async def query(request: QueryRequest):
    query_with_preferences = f"{request.query} considering {request.preferences}"
    start = time.time()
    try:
        res = qa_chain(query_with_preferences)
        answer, docs = res['result'], res['source_documents']
        end = time.time()
        response = {
            "answer": answer,
            "sources": [{ "source": doc.metadata["source"], "content": doc.page_content } for doc in docs]
        }
        return response
    except requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to the server: {e}")
    except Exception as e:
        # Log the exception details and raise an HTTP 500 error
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
