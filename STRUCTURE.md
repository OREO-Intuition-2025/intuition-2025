# Project Structure Documentation

## Overview

This document explains the reorganized structure of the Intuition-2025 Change Management Chat Assistant codebase.

## Directory Structure

```
intuition-2025/
├── app.py                          # Main Streamlit application entry point
├── scraper.py                      # Script to scrape URLs and build knowledge base
├── requirements.txt                # Python dependencies
├── .env.example                    # Template for environment variables
├── .gitignore                      # Git ignore patterns
├── README.md                       # Project documentation
│
├── src/                            # Source code modules
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Configuration and environment variables
│   ├── document_processor.py     # Document loading, splitting, and embedding
│   ├── prompts.py                 # All prompt templates for the RAG system
│   └── rag_workflow.py            # LangGraph RAG workflow implementation
│
├── knowledge-base/                 # Scraped documents (input-*.txt files)
│   ├── input-100.txt
│   ├── input-101.txt
│   └── ...
│
├── cache_db/                       # Cached embeddings (auto-generated)
│   └── sentence-transformers/
│
├── links.txt                       # URLs for scraping
└── intuition-change-management.ipynb  # Analysis notebook

```

## Module Descriptions

### `app.py` - Main Application

- Streamlit UI implementation
- Initializes and caches the RAG system
- Handles chat interface and user interactions
- **Entry point**: Run with `streamlit run app.py`

### `src/config.py` - Configuration

- Centralizes all configuration settings
- Loads environment variables from .env
- Defines paths, model names, and parameters
- **Key settings**:
  - API keys (Google AI, Tavily)
  - Model configurations
  - Chunk sizes and retrieval parameters

### `src/document_processor.py` - Document Processing

- `DocumentProcessor` class handles:
  - Loading documents from knowledge-base/
  - Splitting documents into chunks
  - Creating embeddings with caching
  - Building Chroma vector store
- **Key methods**:
  - `load_documents()`: Load .txt files
  - `split_documents()`: Chunk into pieces
  - `create_vector_store()`: Build searchable index
  - `process_knowledge_base()`: Complete pipeline

### `src/prompts.py` - Prompt Templates

- All LLM prompts in one place for easy modification
- **Prompts included**:
  - `question_router_prompt`: Routes to vectorstore or web search
  - `document_grader_prompt`: Checks document relevance
  - `hallucination_grader_prompt`: Verifies answer grounding
  - `answer_grader_prompt`: Checks answer usefulness
  - `question_rewriter_prompt`: Optimizes queries

### `src/rag_workflow.py` - RAG Workflow

- `RAGWorkflow` class implements LangGraph workflow
- **Node functions**:
  - `retrieve()`: Get documents from vectorstore
  - `generate()`: Create answer using LLM
  - `grade_documents()`: Filter relevant docs
  - `transform_query()`: Rewrite question
  - `web_search()`: Search web as fallback
- **Edge functions** (routing logic):
  - `route_question()`: Choose initial path
  - `decide_to_generate()`: Generate or retry
  - `grade_generation_v_documents_and_question()`: Quality checks

## Workflow Flow

```
User Question
      ↓
[Route Question] → vectorstore OR web_search
      ↓                              ↓
[Retrieve Docs]              [Web Search]
      ↓                              ↓
[Grade Documents]                    ↓
      ↓                              ↓
[Decide] → transform_query           ↓
      ↓         ↓                    ↓
      ←─────────┘                    ↓
      ↓                              ↓
[Generate Answer] ←──────────────────┘
      ↓
[Check Hallucinations & Usefulness]
      ↓
   useful → END
      ↓
not useful/not supported → transform_query or retry
```

## Key Improvements Made

### 1. **Modularization**

- Split monolithic code into focused modules
- Each module has single responsibility
- Easy to test and maintain

### 2. **Configuration Management**

- All settings in one file (`config.py`)
- Environment variables properly loaded
- Easy to adjust parameters

### 3. **Error Handling**

- Proper logging throughout
- Try-catch blocks in critical sections
- User-friendly error messages

### 4. **Code Quality**

- Comprehensive docstrings
- Type hints where appropriate
- PEP 8 compliant formatting
- No code duplication

### 5. **Caching**

- Streamlit `@st.cache_resource` for system init
- Embedding cache for performance
- Avoids reloading on every query

### 6. **Documentation**

- Inline comments explaining logic
- Module-level docstrings
- Function/class documentation
- This structure document

## Usage

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

### Scraping New Data

```bash
# Add URLs to links.txt, then run:
python scraper.py
```

### Development

```python
# Import modules
from src.config import *
from src.document_processor import DocumentProcessor
from src.rag_workflow import RAGWorkflow

# Use components programmatically
processor = DocumentProcessor(...)
workflow = RAGWorkflow(...)
```

## Configuration Options

Edit `src/config.py` to customize:

- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `EMBEDDING_MODEL`: HuggingFace model (default: all-mpnet-base-v2)
- `LLM_MODEL`: Gemini model (default: gemini-2.0-flash)
- `RETRIEVAL_K`: Documents to retrieve (default: 4)
- `WEB_SEARCH_K`: Web results (default: 3)

## Troubleshooting

### Missing API Keys

Error: "GOOGLE_API_KEY not found"

- Solution: Create .env file with API keys

### Import Errors

Error: "No module named 'src'"

- Solution: Run from project root directory

### Embeddings Taking Long

- First run downloads model (~400MB)
- Subsequent runs use cache
- Check cache_db/ directory

### Streamlit Not Found

Error: "streamlit: command not found"

- Solution: `pip install streamlit`

## Future Enhancements

Potential improvements:

1. Add unit tests for each module
2. Implement conversation memory
3. Add more change management models
4. Create REST API alongside UI
5. Add user authentication
6. Implement feedback mechanism
7. Support multiple document formats
8. Add visualization of RAG workflow

## Contributing

When adding features:

1. Keep modules focused and small
2. Add docstrings to all functions
3. Update this documentation
4. Test thoroughly before committing
