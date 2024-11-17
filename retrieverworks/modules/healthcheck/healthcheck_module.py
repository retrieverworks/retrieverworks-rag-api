# -*- coding: utf-8 -*-
"""
Author: Mihai Criveti
Description: Health Check Module for Retrieverworks
Provides a simple health check endpoint to verify service availability.
"""

from fastapi import APIRouter

# Create a router for the health check module
router = APIRouter()

@router.get("/health", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint to verify the service is running.
    Returns a simple status message and a 200 OK response.
    """
    return {
        "status": "healthy",
        "message": "Retrieverworks is operational.",
    }

def add_api_module(app):
    """
    Register the health check routes with the main FastAPI application.

    Args:
        app (FastAPI): The main application instance.
    """
    app.include_router(router, prefix="/api")
