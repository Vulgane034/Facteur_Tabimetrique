"""
FastAPI application entry point and configuration.

Provides the main FastAPI application with CORS middleware, routing,
and utility endpoints for API information and health checks.

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

import logging
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings, logger
from app.api.routes import router
from app.services.storage import model_storage


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Initializes FastAPI with CORS middleware, routes, and utility endpoints.
    
    Returns:
        Configured FastAPI application instance with all middleware and routes
    """
    
    # Create FastAPI instance with metadata
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "Advanced Variable Selection Method using Facteur Tabimetrique (FT). "
            "Combines Pearson (ζ), Kendall (τ), Distance Correlation (dCor), "
            "and adaptive weighting via MLP to compute comprehensive feature importance scores."
        ),
        contact={
            "name": "EYAGA TABI Jean François Régis",
            "email": "francoistabi294@gmail.com",
            "url": "https://github.com/vulgane034",
        },
        license_info={
            "name": "MIT License",
        },
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    
    logger.info("FastAPI application created")
    
    # ========================================================================
    # CORS Middleware
    # ========================================================================
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    logger.info(f"CORS middleware configured with origins: {settings.cors_origins}")
    
    # ========================================================================
    # Root Endpoints
    # ========================================================================
    
    @app.get(
        "/",
        status_code=200,
        summary="API Information",
        description="Returns API metadata and author contact information",
    )
    async def root() -> Dict:
        """Root endpoint providing API information.
        
        Returns comprehensive metadata about the API including author details,
        version, and quick links to documentation.
        
        Returns:
            Dictionary with API info, version, author, and documentation links
        """
        logger.debug("GET /: Root endpoint accessed")
        
        return {
            "status": "online",
            "service": "Facteur Tabimetrique API",
            "version": settings.api_version,
            "title": settings.api_title,
            "description": (
                "Advanced variable selection method combining Pearson (ζ), Kendall (τ), "
                "Distance Correlation (dCor), and transitive dependence (C) with "
                "adaptive MLP-based weighting."
            ),
            "author": {
                "name": "EYAGA TABI Jean François Régis",
                "email": "francoistabi294@gmail.com",
                "github": "https://github.com/vulgane034",
                "linkedin": "https://www.linkedin.com/in/francois-tabi-03a4b7235",
            },
            "endpoints": {
                "documentation": "http://localhost:8000/api/docs",
                "redoc": "http://localhost:8000/api/redoc",
                "openapi": "http://localhost:8000/api/openapi.json",
                "health": "http://localhost:8000/health",
                "api_v1": "http://localhost:8000/api/v1",
            },
            "features": [
                "Model training with MLP weight learning",
                "Feature importance scoring",
                "Feature selection with threshold",
                "Method comparison (Pearson, Spearman, Distance Correlation)",
                "LRU-based model storage",
                "Thread-safe operations",
                "CSV data upload",
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    # ========================================================================
    # Health Check Endpoints
    # ========================================================================
    
    @app.get(
        "/health",
        status_code=200,
        summary="API Health Check",
        description="Returns API status and storage statistics",
    )
    async def health_check() -> Dict:
        """API health check endpoint.
        
        Provides comprehensive status information including storage statistics,
        model count, and API availability.
        
        Returns:
            Dictionary with health status and storage metrics
        """
        try:
            logger.debug("GET /health: Health check requested")
            
            stats = model_storage.get_stats()
            
            response = {
                "status": "healthy",
                "service": "Facteur Tabimetrique API",
                "version": settings.api_version,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_info": "Service running",
                "storage": {
                    "models_loaded": stats["total_models"],
                    "max_capacity": stats["max_models"],
                    "available_slots": stats["available_slots"],
                    "utilization_percent": stats["utilization"],
                    "oldest_model": stats["oldest_model"],
                    "newest_model": stats["newest_model"],
                },
                "database": "in-memory (no persistence)",
            }
            
            logger.debug(f"GET /health: ✓ Health check passed (models: {stats['total_models']}/{stats['max_models']})")
            return response
            
        except Exception as e:
            logger.error(f"GET /health: ✗ Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    # ========================================================================
    # Documentation Redirects
    # ========================================================================
    
    @app.get(
        "/docs",
        include_in_schema=False,
        summary="Swagger UI Documentation",
    )
    async def redirect_docs():
        """Redirect to Swagger UI documentation.
        
        Returns:
            Redirect response to /api/docs
        """
        logger.debug("GET /docs: Redirecting to Swagger UI")
        return RedirectResponse(url="/api/docs")
    
    @app.get(
        "/redoc",
        include_in_schema=False,
        summary="ReDoc Documentation",
    )
    async def redirect_redoc():
        """Redirect to ReDoc documentation.
        
        Returns:
            Redirect response to /api/redoc
        """
        logger.debug("GET /redoc: Redirecting to ReDoc")
        return RedirectResponse(url="/api/redoc")
    
    # ========================================================================
    # API Router
    # ========================================================================
    
    app.include_router(router)
    logger.info(f"API router included with prefix: /api/v1")
    
    # ========================================================================
    # Startup Events
    # ========================================================================
    

    @app.on_event("startup")
    async def startup_event() -> None:
        """Run on application startup."""
        logger.info(f"Starting {settings.api_title} v{settings.api_version}")
        logger.info(f"Debug mode: {settings.debug}")
        logger.info(f"CORS origins: {settings.cors_origins}")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Run on application shutdown."""
        logger.info(f"Shutting down {settings.api_title}")
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn  # type: ignore
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
