# -------------------------------
#  IMPORTS
# -------------------------------
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_ollama import ChatOllama
import hashlib


# -------------------------------
#  STEP 1: Helper - Extract YouTube Video ID
# -------------------------------
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format.")


# -------------------------------
#  STEP 2: Fetch Transcript
# -------------------------------
def get_transcript(video_url):
    video_id = extract_video_id(video_url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])


# -------------------------------
#  STEP 3: Chunk Transcript
# -------------------------------
def chunk_transcript(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


# -------------------------------
# STEP 4: Create Unique Collection Name
# -------------------------------
def generate_collection_name(video_url):
    return hashlib.md5(video_url.encode()).hexdigest()


# -------------------------------
# STEP 5: Embed + Store in ChromaDB (skip if exists)
# -------------------------------
def store_transcript(video_url, chunks, embedder):
    collection_name = generate_collection_name(video_url)
    persist_dir = "youtube_chroma_db"

    client = chromadb.Client(Settings(persist_directory=persist_dir))
    try:
        collection = client.get_collection(collection_name)
        print("✅ Vector store already exists. Skipping embedding.")
    except:
        collection = client.create_collection(collection_name)
        embeddings = embedder.encode(chunks)
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                embeddings=[embeddings[i]],
                ids=[str(i)]
            )
        print("✅ Chunks embedded and stored.")
    return collection


# -------------------------------
#  STEP 6: Ask Question via LLM with Context
# -------------------------------
def ask_question(question, collection, embedder):
    question_embedding = embedder.encode([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )

    retrieved_chunks = [doc for doc in results["documents"][0]]

    context = "\n".join(retrieved_chunks)

    prompt = (
        "You are a helpful assistant. Use the provided transcript context to answer the question clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    llm = ChatOllama(model="mistral")
    answer = llm.invoke(prompt)
    return answer


# -------------------------------
#  MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    # Enter YouTube Link & Question
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    user_question = "What is this video about?"

    # Step 1: Get transcript
    print("📥 Fetching transcript...")
    transcript = get_transcript(video_url)

    # Step 2: Chunk transcript
    print("✂️ Chunking transcript...")
    chunks = chunk_transcript(transcript)

    # Step 3: Load embedding model
    print("📐 Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 4: Store in ChromaDB
    print("📦 Storing in ChromaDB...")
    collection = store_transcript(video_url, chunks, embedder)

    # Step 5: Ask question
    print("❓ Asking question...")
    answer = ask_question(user_question, collection, embedder)

    # Output answer
    print("\n🧠 Answer:\n", answer)
