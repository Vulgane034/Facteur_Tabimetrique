#!/usr/bin/env python3
"""Simple test to verify all modules import correctly."""

import sys
import traceback

def test_imports():
    """Test all critical imports."""
    tests = [
        ("FastAPI", "from fastapi import FastAPI"),
        ("Uvicorn", "import uvicorn"),
        ("Pydantic", "from pydantic import BaseModel"),
        ("Pandas", "import pandas as pd"),
        ("NumPy", "import numpy as np"),
        ("SciPy", "from scipy import stats"),
        ("TensorFlow", "import tensorflow as tf"),
        ("App Config", "from app.config import Settings"),
        ("Models", "from app.models.requests import TrainRequest"),
        ("Storage", "from app.services.storage import model_storage"),
        ("FT Service", "from app.services.ft_service import ft_service"),
        ("Routes", "from app.api.routes import router"),
        ("Main App", "from app.main import app"),
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("TESTING IMPORTS")
    print("=" * 70)
    
    for name, code in tests:
        try:
            exec(code)
            print(f"✓ {name:30} OK")
            passed += 1
        except Exception as e:
            print(f"✗ {name:30} FAILED: {str(e)[:40]}")
            failed += 1
            traceback.print_exc()
    
    print("=" * 70)
    print(f"RESULT: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
