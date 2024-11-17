# -*- coding: utf-8 -*-
"""
Author: Mihai Criveti
Description: Retrieverworks: Document models and Metadata
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Common model for document metadata"""
    filename: str = Field(..., description="Original filename")
    stored_filename: str = Field(..., description="Generated UUID filename")
    content_type: str = Field(..., description="MIME type of the document")
    size: int = Field(..., description="Document size in bytes")
    timestamp: datetime = Field(default_factory=datetime.now)
    path: str = Field(..., description="Public access path")
    description: Optional[str] = None

    model_config = {
      "json_schema_extra": {
          "example": {
              "filename": "example_document.docx",
              "stored_filename": "123e4567-e89b-12d3-a456-426614174000.docx",
              "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
              "size": 456789,
              "timestamp": "2024-11-17T10:15:30.123456",
              "path": "/public/123e4567-e89b-12d3-a456-426614174000.docx",
              "description": "A sample Word document."
          }
      }
  }



class DocumentResponse(BaseModel):
    """Common response model for document operations"""
    message: str
    document: DocumentMetadata
