# backend_with_chain.py

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from chromadb.config import Settings
import hashlib
import os


# Extract video ID from URL
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")


# Get transcript
def get_transcript(url):
    video_id = extract_video_id(url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([seg["text"] for seg in transcript])


# Hash URL to use as Chroma collection name
def get_collection_name(url):
    return hashlib.md5(url.encode()).hexdigest()


# Process transcript into vectorstore
def get_vectorstore(url, transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection_name = get_collection_name(url)

    persist_directory = "youtube_chroma_chain"
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    vectorstore.persist()
    return vectorstore


# Build QA chain using LangChain
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model="mistral")
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain


# Top-level function to run everything
def get_answer_from_video(url, question):
    transcript = get_transcript(url)
    vectorstore = get_vectorstore(url, transcript)
    qa_chain = build_qa_chain(vectorstore)
    result = qa_chain.invoke({"query": question})
    return result["result"]
