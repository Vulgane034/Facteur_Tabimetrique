"""
Startup script for Facteur Tabimetrique API.

Launches the FastAPI server with Uvicorn using configuration from settings.
Displays banner with API information and author contact details.

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

import os
import sys

import uvicorn

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import settings
from app.config import settings


def main() -> None:
    """Start the Facteur Tabimetrique API server.
    
    Displays startup banner with API metadata and author information,
    then launches Uvicorn server with configuration from settings.
    """
    
    # Display startup banner
    print("\n" + "=" * 80)
    print("  🚀 FACTEUR TABIMETRIQUE API - Advanced Variable Selection Method")
    print("=" * 80)
    print()
    print(f"  Service Name   : {settings.api_title}")
    print(f"  Version        : {settings.api_version}")
    print(f"  Author         : EYAGA TABI Jean François Régis")
    print(f"  Email          : francoistabi294@gmail.com")
    print(f"  GitHub         : https://github.com/vulgane034")
    print(f"  LinkedIn       : https://www.linkedin.com/in/francois-tabi-03a4b7235")
    print()
    print(f"  API Server     : http://{settings.api_host}:{settings.api_port}")
    print(f"  Documentation  : http://{settings.api_host}:{settings.api_port}/api/docs")
    print(f"  ReDoc          : http://{settings.api_host}:{settings.api_port}/api/redoc")
    print()
    print(f"  CORS Origins   : {', '.join(settings.cors_origins)}")
    print(f"  Log Level      : {settings.log_level}")
    print(f"  Reload Mode    : {'ON (development)' if settings.debug else 'OFF (production)'}")
    print(f"  Storage Max    : {settings.model_storage_limit} models")
    print()
    print("=" * 80)
    print("  Starting API server...")
    print("=" * 80)
    print()
    
    try:
        # Launch Uvicorn server
        uvicorn.run(
            "app.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.debug,
            log_level=settings.log_level.lower(),
            access_log=True,
            use_colors=True,
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("  ⛔ API server stopped by user")
        print("=" * 80 + "\n")
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"  ❌ API server failed with error: {str(e)}")
        print("=" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

