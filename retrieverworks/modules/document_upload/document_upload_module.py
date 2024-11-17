# -*- coding: utf-8 -*-
"""
Author: Mihai Criveti
Description: Document Upload Module for Retrieverworks
Provides endpoints for uploading and listing documents.
"""

import logging
import mimetypes
import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette import status
from retrieverworks.document_models import DocumentMetadata, DocumentResponse

# Configure logging
log = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

async def save_upload_document(
    file: UploadFile,
    description: Optional[str],
    public_dir: Path
) -> DocumentMetadata:
    """Save uploaded document and create metadata"""
    try:
        # Generate UUID filename
        document_uuid = str(uuid4())
        ext = mimetypes.guess_extension(file.content_type) or Path(file.filename).suffix
        stored_filename = f"{document_uuid}{ext}"
        document_path = public_dir / stored_filename

        # Save document
        size = 0
        with open(document_path, 'wb') as f:
            while chunk := await file.read(8192):
                size += len(chunk)
                f.write(chunk)

        # Create metadata
        metadata = DocumentMetadata(
            filename=file.filename,
            stored_filename=stored_filename,
            content_type=file.content_type,
            size=size,
            path=f"/public/{stored_filename}",
            description=description
        )

        # Save metadata as a JSON file next to the document
        metadata_path = public_dir / f"{document_uuid}-metadata.json"
        with open(metadata_path, 'w') as metadata_file:
            json.dump(metadata.dict(), metadata_file, indent=2, default=str)

        return metadata

    except Exception as e:
        log.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving document: {str(e)}"
        )

@router.post(
    "/document_upload",
    tags=["Document Management"],
    response_model=DocumentResponse,
    summary="Upload document",
    description="Upload a document and store its metadata"
)
async def document_upload(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
) -> DocumentResponse:
    """Handle document upload"""
    try:
        public_dir = Path("public")
        public_dir.mkdir(exist_ok=True)

        metadata = await save_upload_document(
            file,
            description,
            public_dir
        )

        return DocumentResponse(
            message="Document uploaded successfully",
            document=metadata
        )

    except Exception as e:
        log.error(f"Error in document_upload: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get(
    "/documents",
    tags=["Document Management"],
    summary="List uploaded documents",
    description="List all uploaded documents and their metadata"
)
async def list_documents() -> list[DocumentMetadata]:
    """List all uploaded documents and their metadata"""
    try:
        public_dir = Path("public")
        public_dir.mkdir(exist_ok=True)

        documents = []
        for metadata_file in public_dir.glob("*-metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                documents.append(metadata)

        return documents

    except Exception as e:
        log.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

def add_api_module(app):
    """
    Register the document upload routes with the main FastAPI application.

    Args:
        app (FastAPI): The main application instance.
    """
    app.include_router(router, prefix="/api")
