"""
Invoice OCR API - Main Application Entry Point.

A FastAPI-based backend for vendor-specific invoice OCR with ML-powered
field extraction. The system learns from validated invoices to improve
accuracy over time.

Features:
- Tesseract OCR for text extraction from images/PDFs
- Position-based field extraction using learned vendor templates
- Pattern learning from validated invoices
- KNN-based inference for small datasets
- PyTorch neural network model for field prediction
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.routes import router
from config import config

# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="Invoice OCR API",
    description="Vendor-specific invoice OCR with ML training",
    version="1.0.0"
)

# =============================================================================
# CORS (Cross-Origin Resource Sharing) Configuration
# Allows the React frontend to communicate with this API
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # Allow all origins for development
    allow_credentials=True,           # Allow cookies/auth headers
    allow_methods=["*"],              # Allow all HTTP methods
    allow_headers=["*"],
)

# =============================================================================
# Static File Serving
# Mount the uploads directory to serve invoice images via HTTP
# =============================================================================

uploads_dir = Path(config['app']['upload_dir'])
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# Include API routes with /api prefix
app.include_router(router)


# =============================================================================
# Application Lifecycle Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Initialize application on startup.

    Creates necessary storage directories if they don't exist:
    - Upload directory for temporary invoice images
    - Vendor storage directory for vendor-specific data (models, patterns, training data)
    - Invoice storage directory for processed invoice records
    """
    Path(config['app']['upload_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['vendors']['storage_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['invoices']['storage_dir']).mkdir(parents=True, exist_ok=True)
    print("Invoice OCR API started successfully")


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/")
async def root():
    """
    Root endpoint providing API metadata and links to documentation.

    Returns:
        dict: API information including available endpoints
    """
    return {
        "message": "Invoice OCR API",
        "docs": "/docs",
        "health": "/api/health"
    }


# =============================================================================
# Development Server Runner
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config['app']['host'],
        port=config['app']['port'],
        reload=config['app']['debug']
    )
