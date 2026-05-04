import os
import hashlib
import torch
import chromadb
import shutil
from paths import VECTOR_DB_DIR
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import load_all_knowledge_base


def initialize_db(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "knowledge_base",
    delete_existing: bool = False,
) -> chromadb.Collection:
    """
    Initialize a ChromaDB instance and persist it to disk.

    Args:
        persist_directory (str): The directory where ChromaDB will persist data. Defaults to "./vector_db"
        collection_name (str): The name of the collection to create/get. Defaults to "knowledge_base"
        delete_existing (bool): Whether to delete the existing database if it exists. Defaults to False
    Returns:
        chromadb.Collection: The ChromaDB collection instance
    """
    if os.path.exists(persist_directory) and delete_existing:
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=persist_directory)

    # Create or get a collection
    try:
        # Try to get existing collection first
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection: {collection_name}")
    except Exception:
        # If collection doesn't exist, create it
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:batch_size": 10000,
            },  # Use cosine distance for semantic search
        )
        print(f"Created new collection: {collection_name}")

    print(f"ChromaDB initialized with persistent storage at: {persist_directory}")

    return collection


def get_db_collection(
    persist_directory: str = VECTOR_DB_DIR,
    collection_name: str = "knowledge_base",
) -> chromadb.Collection:
    """
    Get a ChromaDB client instance.

    Args:
        persist_directory (str): The directory where ChromaDB persists data
        collection_name (str): The name of the collection to get

    Returns:
        chromadb.PersistentClient: The ChromaDB client instance
    """
    return chromadb.PersistentClient(path=persist_directory).get_collection(
        name=collection_name
    )


def chunk_knowledge(
    knowledge: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    Chunk the knowledge into smaller documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_text(knowledge)


def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Embed documents using a model.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    embeddings = model.embed_documents(documents)
    return embeddings


def insert_knowledge_base(collection: chromadb.Collection, knowledge_base: list[tuple[str, str]]):
    """
    Insert documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        knowledge_base (list[tuple[str, str]]): List of (content, source_name) tuples

    Returns:
        None
    """
    next_id = collection.count()

    for knowledge, source_name in knowledge_base:
        chunked_knowledge = chunk_knowledge(knowledge)
        embeddings = embed_documents(chunked_knowledge)
        ids = [f"document_{i}" for i in range(next_id, next_id + len(chunked_knowledge))]
        metadatas = [{"source": source_name} for _ in chunked_knowledge]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunked_knowledge,
            metadatas=metadatas,
        )
        next_id += len(chunked_knowledge)


def compute_file_hash(content: str) -> str:
    """Return a SHA-256 hex digest of the file content string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_indexed_hash(collection: chromadb.Collection, source_name: str) -> str | None:
    """
    Return the stored file hash for a given source, or None if not indexed.
    The hash is stored in the metadata of the first chunk of the document.
    """
    results = collection.get(where={"source": source_name}, limit=1, include=["metadatas"])
    if results["ids"]:
        return results["metadatas"][0].get("file_hash")
    return None


def delete_chunks_by_source(collection: chromadb.Collection, source_name: str) -> int:
    """
    Delete all chunks belonging to a given source from the collection.
    Returns the number of chunks deleted.
    """
    results = collection.get(where={"source": source_name}, include=[])
    ids_to_delete = results["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def upsert_document(
    collection: chromadb.Collection,
    content: str,
    source_name: str,
) -> str:
    """
    Intelligently index a single document:
    - If the file is new: index it.
    - If the file already exists and is unchanged: skip it.
    - If the file already exists and has changed: delete old chunks and reindex.

    Returns a status string: "added", "updated", or "unchanged".
    """
    new_hash = compute_file_hash(content)
    existing_hash = get_indexed_hash(collection, source_name)

    if existing_hash is None:
        status = "added"
    elif existing_hash == new_hash:
        return "unchanged"
    else:
        delete_chunks_by_source(collection, source_name)
        status = "updated"

    chunks = chunk_knowledge(content)
    embeddings = embed_documents(chunks)

    # Use a source+index scheme for IDs so they don't clash with other sources
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "file_hash": new_hash} for _ in chunks]

    collection.add(
        embeddings=embeddings,
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )
    return status


def list_indexed_sources(collection: chromadb.Collection) -> list[dict]:
    """
    Return a list of all unique indexed sources with their chunk counts.
    """
    results = collection.get(include=["metadatas"])
    source_info: dict[str, dict] = {}
    for meta in results["metadatas"]:
        src = meta.get("source", "unknown")
        if src not in source_info:
            source_info[src] = {"source": src, "chunks": 0, "file_hash": meta.get("file_hash", "")}
        source_info[src]["chunks"] += 1
    return list(source_info.values())


def main():
    collection = initialize_db(
        persist_directory=VECTOR_DB_DIR,
        collection_name="knowledge_base",
        delete_existing=True,
    )
    knowledge_base = load_all_knowledge_base()
    insert_knowledge_base(collection, knowledge_base)

    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()
