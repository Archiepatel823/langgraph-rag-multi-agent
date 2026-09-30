from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from llms import embeddings
import config


def index_pdf(file_path: str) -> int:
    """Load a PDF, split it into chunks, embed them and store them in Pinecone."""

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError("PDF contains no readable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Could not create document chunks.")

    # Add useful metadata
    for chunk in chunks:
        chunk.metadata["source"] = Path(file_path).name

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX,
        pinecone_api_key=config.PINECONE_API_KEY,
    )

    return len(chunks)