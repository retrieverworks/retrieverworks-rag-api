# -*- coding: utf-8 -*-
"""
Author: Mihai Criveti
Description: Retrieverworks: Modular Retriever Augmented Generation
"""

import logging
from importlib import import_module
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - retrieverworks.api - %(levelname)s - %(message)s'
)
log = logging.getLogger("retrieverworks.api")

# Load environment variables from .env file
load_dotenv()

# Configure storage paths
STORAGE_DIR = Path("public")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Set maximum upload size (100MB)
MAX_UPLOAD_SIZE = 100 * 1024 * 1024

# Set up FastAPI application
app = FastAPI(
    title="Retrieverworks: Modular Retriever Augmented Generation",
    version="1.0",
    description="A modular retriever-augmented generation service",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def register_api_modules_from_folder(folder: Path) -> None:
    """
    Register API modules from a specified folder.

    This function dynamically imports modules with "_module.py" suffix from the
    given folder and registers their routes with the FastAPI application.

    Args:
        folder (Path): The directory containing module files.
    """
    log.info(f"Registering API modules from: {folder}")
    for file in folder.rglob("*_module.py"):
        # Construct the module path relative to the retrieverworks package
        relative_path = file.relative_to(folder.parent).with_suffix("")
        module_name = f"retrieverworks.{'.'.join(relative_path.parts)}"
        try:
            module = import_module(module_name)
            log.info(f"Loaded module: {module_name}")
            if hasattr(module, "add_api_module"):
                module.add_api_module(app)
                log.info(f"Registered API module: {module_name}")
        except ModuleNotFoundError as e:
            log.error(f"Failed to load API module: {module_name}. Error: {e}")

# Register API modules from the 'retrieverworks/modules' folder
modules_folder = Path(__file__).parent / "modules"
register_api_modules_from_folder(modules_folder)

# Log available endpoints
for route in app.routes:
    if hasattr(route, "methods"):
        log.info(f"API available at: {route.path} [{', '.join(route.methods)}]")
    else:
        log.info(f"Static or mounted route available at: {route.path}")

# Log startup information
log.info(f"Storage directory: {STORAGE_DIR.absolute()}")
log.info(f"Maximum upload size: {MAX_UPLOAD_SIZE / (1024 * 1024):.2f} MB")
log.info("API documentation: http://localhost:8080/docs")

if __name__ == "__main__":
    import uvicorn

    # Configure uvicorn server
    uvicorn.run(
        "retrieverworks.api:app",  # Explicitly specify the app instance
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=True,  # Enable auto-reload during development
    )
