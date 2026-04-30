#!/usr/bin/env python3
"""LexRAG API server entry point."""

import uvicorn
from config import settings
from lexrag.api.server import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.api_reload,
        log_level=settings.api_log_level,
    )
