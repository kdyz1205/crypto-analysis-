"""Vercel serverless entry point — wraps the FastAPI app."""
import sys
import os

# Add project root to path so server package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.app import app  # noqa: E402
