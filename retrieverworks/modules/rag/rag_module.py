# -*- coding: utf-8 -*-
"""
Author: Mihai Criveti
Description: RAG Operations Module for Retrieverworks
Provides endpoints for text splitting, document conversion, embeddings, and querying
with support for multiple LLM backends.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Literal
from uuid import uuid4
from abc import ABC, abstractmethod

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
import requests
import openai

# Configure logging
log = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Constants
CHUNK_SIZE = 500
CHROMA_DB_DIR = "./chromadb"
COLLECTION_NAME = "documents"

# LLM Configuration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "granite3-dense"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"

# Pydantic models for requests/responses
class TextSplitRequest(BaseModel):
    text: str
    chunk_size: Optional[int] = CHUNK_SIZE

class TextSplitResponse(BaseModel):
    chunks: List[str]
    chunk_count: int

class VectorSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3

class VectorSearchResponse(BaseModel):
    matches: List[str]
    scores: Optional[List[float]]

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 3
    llm_backend: Literal["ollama", "chatgpt", "fakellm"] = "fakellm"
    model_name: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    context: List[str]
    llm_backend: str
    model_name: str

# LLM Backend Abstract Base Class
class LLMBackend(ABC):
    @abstractmethod
    def generate(self, query: str, context: List[str]) -> str:
        pass

# Ollama Backend Implementation
class OllamaBackend(LLMBackend):
    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL):
        self.model_name = model_name
        self.endpoint = OLLAMA_ENDPOINT

    def generate(self, query: str, context: List[str]) -> str:
        try:
            prompt = f"Context: {' '.join(context)}\n\nQuestion: {query}"
            response = requests.post(self.endpoint, json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            })
            response.raise_for_status()
            return response.json().get("response", "No response received from Ollama.")
        except Exception as e:
            log.error(f"Ollama query error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to query Ollama: {str(e)}"
            )

# ChatGPT Backend Implementation
class ChatGPTBackend(LLMBackend):
    def __init__(self, model_name: str = DEFAULT_OPENAI_MODEL):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        self.model_name = model_name
        openai.api_key = OPENAI_API_KEY

    def generate(self, query: str, context: List[str]) -> str:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": f"Context: {' '.join(context)}\n\nQuestion: {query}"}
            ]
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"ChatGPT query error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to query ChatGPT: {str(e)}"
            )

# Fake LLM Backend for Testing
class FakeLLMBackend(LLMBackend):
    def __init__(self, model_name: str = "fake-model"):
        self.model_name = model_name

    def generate(self, query: str, context: List[str]) -> str:
        return f"Fake response to query: '{query}' based on {len(context)} context chunks."

# LLM Factory
def get_llm_backend(backend_type: str, model_name: Optional[str] = None) -> LLMBackend:
    if backend_type == "ollama":
        return OllamaBackend(model_name or DEFAULT_OLLAMA_MODEL)
    elif backend_type == "chatgpt":
        return ChatGPTBackend(model_name or DEFAULT_OPENAI_MODEL)
    elif backend_type == "fakellm":
        return FakeLLMBackend(model_name or "fake-model")
    else:
        raise ValueError(f"Unsupported LLM backend: {backend_type}")

# Utility functions
def split_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks of specified size."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return [chunk for chunk in chunks if chunk.strip()]

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text content from PDF file."""
    try:
        reader = PdfReader(str(file_path))
        return " ".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        log.error(f"PDF extraction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from PDF: {str(e)}"
        )

def get_chromadb_collection():
    """Get or create ChromaDB collection."""
    try:
        client = chromadb.Client(Settings(
            persist_directory=CHROMA_DB_DIR,
            is_persistent=True
        ))
        return client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        log.error(f"ChromaDB error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"ChromaDB operation failed: {str(e)}"
        )

# API Routes
@router.post(
    "/split",
    response_model=TextSplitResponse,
    tags=["RAG Operations"],
    summary="Split text into chunks"
)
async def split_text_route(request: TextSplitRequest) -> TextSplitResponse:
    """Split input text into chunks of specified size."""
    try:
        chunks = split_text(request.text, request.chunk_size)
        return TextSplitResponse(chunks=chunks, chunk_count=len(chunks))
    except Exception as e:
        log.error(f"Text splitting error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post(
    "/convert/pdf",
    tags=["RAG Operations"],
    summary="Convert PDF to text"
)
async def convert_pdf_route(file: UploadFile) -> dict:
    """Convert uploaded PDF to text."""
    try:
        # Save uploaded file
        file_path = Path("public") / f"{uuid4()}.pdf"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract text
        text = extract_text_from_pdf(file_path)
        
        # Clean up
        file_path.unlink()

        return {
            "text": text,
            "character_count": len(text)
        }
    except Exception as e:
        log.error(f"PDF conversion error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post(
    "/embed",
    tags=["RAG Operations"],
    summary="Generate embeddings and store in vector DB"
)
async def embed_text_route(chunks: List[str]) -> dict:
    """Generate embeddings for text chunks and store in ChromaDB."""
    try:
        collection = get_chromadb_collection()
        
        # Generate unique IDs for chunks
        chunk_ids = [f"chunk_{uuid4()}" for _ in chunks]
        
        # Add to ChromaDB
        collection.add(
            documents=chunks,
            ids=chunk_ids
        )
        
        return {
            "message": f"Successfully embedded {len(chunks)} chunks",
            "chunk_ids": chunk_ids
        }
    except Exception as e:
        log.error(f"Embedding error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post(
    "/retrieve",
    response_model=VectorSearchResponse,
    tags=["RAG Operations"],
    summary="Retrieve similar texts from vector DB"
)
async def retrieve_route(request: VectorSearchRequest) -> VectorSearchResponse:
    """Retrieve similar texts from ChromaDB."""
    try:
        collection = get_chromadb_collection()
        results = collection.query(
            query_texts=[request.query],
            n_results=request.max_results
        )
        
        return VectorSearchResponse(
            matches=results["documents"][0],
            scores=results.get("distances", [None])[0]
        )
    except Exception as e:
        log.error(f"Retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG Operations"],
    summary="Query documents using RAG with configurable LLM backend"
)
async def query_route(request: QueryRequest) -> QueryResponse:
    """Query documents using retrieval-augmented generation with specified LLM backend."""
    try:
        # Get LLM backend
        llm = get_llm_backend(request.llm_backend, request.model_name)
        
        # Retrieve relevant contexts
        collection = get_chromadb_collection()
        results = collection.query(
            query_texts=[request.query],
            n_results=request.max_results
        )
        contexts = results["documents"][0]
        
        # Generate answer using selected LLM
        answer = llm.generate(request.query, contexts)
        
        return QueryResponse(
            answer=answer,
            context=contexts,
            llm_backend=request.llm_backend,
            model_name=llm.model_name
        )
    except Exception as e:
        log.error(f"Query error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

def add_api_module(app):
    """
    Register the RAG operations routes with the main FastAPI application.

    Args:
        app (FastAPI): The main application instance.
    """
    app.include_router(router, prefix="/api/rag")