"""
Configuration settings for the Change Management Chat Assistant.
All environment variables and constants are defined here.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Paths
KNOWLEDGE_BASE_DIR = "./knowledge-base"
CACHE_DB_DIR = "./cache_db"

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "gemini-2.0-flash"

# Text Splitting Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  # Increased from 20 for better context retention

# Retrieval Configuration
RETRIEVAL_K = 4  # Number of documents to retrieve
WEB_SEARCH_K = 3  # Number of web search results

# UI Configuration
APP_TITLE = "Metamorphosis"
APP_SUBTITLE = "Chat Assistant for Change Management"
