from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

import fastapi
from fastapi import FastAPI

import pydantic
from pydantic import BaseModel

load_dotenv()

def initialise_chain():
    loader = DirectoryLoader(
        "./knowledge-base",
        glob="**/*.txt",
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(docs)

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    store = LocalFileStore("./cache_db")

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(embeddings_model,store,namespace=embeddings_model.model_name)

    db = Chroma.from_documents(texts, cached_embedder)

    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )

    return qa

chatbot  = initialise_chain()
app = FastAPI()

class Data(BaseModel):
    content:str

@app.get("/")
async def root():
    return "Hello, World!"

@app.post("/chat")
async def chat(query:Data):
    result = Data()
    result.content = chatbot.invoke(query.content)["result"]
    return result