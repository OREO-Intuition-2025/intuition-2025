# Intuition-2025: Chat Assistant for Change Management

A specialized AI-powered chat assistant designed to provide expert guidance on change management strategies, models, and best practices. Built using Retrieval-Augmented Generation (RAG) with a knowledge base of academic papers and resources on change management frameworks like Lewin's Model, Kotter's 8-Step, McKinsey 7S, and ADKAR.

## Features

- **Interactive Chat Interface**: Streamlit-based UI for natural language Q&A about change management.
- **RAG-Powered Responses**: Retrieves relevant information from a curated knowledge base of academic papers and articles, ensuring grounded and accurate answers.
- **Intelligent Routing**: Automatically routes queries to vector search (for change management topics) or web search (for general queries).
- **Quality Assurance**: Includes document relevance grading, hallucination detection, and answer usefulness checks.
- **Model Recommendation**: Exploratory analysis in the notebook for recommending change management models based on contextual factors (urgency, complexity, resistance, change level).
- **Data Pipeline**: Automated scraping of academic resources to build and update the knowledge base.

## Installation

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - Google Generative AI (Gemini model)
  - Tavily (web search)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/OREO-Intuition-2025/intuition-2025.git
   cd intuition-2025
   ```

2. Install dependencies:

   ```bash
   pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-google-genai langchain-chroma langgraph streamlit sentence-transformers beautifulsoup4 requests python-dotenv tavily-python
   ```

3. Create a `.env` file in the root directory and add your API keys:

   ```
   GOOGLE_API_KEY=your_google_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

4. (Optional) For notebook analysis: Install additional packages:
   ```bash
   pip install pandas scikit-learn transformers kaggle
   ```

## Usage

### Running the Chat Assistant

1. Populate the knowledge base (if not already done):

   ```bash
   python scraper.py
   ```

   This scrapes content from `links.txt` and saves to `knowledge-base/`.

2. Launch the app:
   ```bash
   streamlit run app.py
   ```
   Open the provided URL in your browser. Ask questions like "What is Lewin's change model?" or "How to handle resistance in organizational change?"

### Running Analysis (Notebook)

Open `intuition-change-management.ipynb` in Jupyter and run cells to:

- Analyze change management models via web scraping.
- Recommend models based on role and change context (e.g., "Role: Engineer. Change: AI implementation").
- Fine-tune models on HR datasets for strategy prediction.

## Project Structure

```
intuition-2025/
├── app.py                 # Main Streamlit app with RAG pipeline
├── scraper.py             # Script to scrape and build knowledge base
├── intuition-change-management.ipynb  # Notebook for model analysis and recommendations
├── links.txt              # List of URLs for scraping
├── README.md              # This file
├── knowledge-base/        # Scraped text files (input-*.txt)
├── cache_db/              # Cached embeddings for performance
└── .env                   # Environment variables (API keys)
```

## How It Works

1. **Data Collection**: `scraper.py` fetches content from academic URLs in `links.txt` using BeautifulSoup, cleaning and storing text in `knowledge-base/`.
2. **Preprocessing**: `app.py` loads text, splits into chunks, generates embeddings (cached in `cache_db/`), and indexes in Chroma vector database.
3. **Query Processing**: User queries are routed via LangGraph:
   - Vectorstore retrieval for change management topics.
   - Web search via Tavily for others.
   - Documents are graded for relevance; queries rewritten if needed.
4. **Generation**: Gemini 2.0 Flash generates answers, with checks for grounding and usefulness.
5. **Analysis**: The notebook uses NLP (TF-IDF, Sentence Transformers) and ML (fine-tuning) to score and recommend models based on factors like urgency and resistance.

## Dependencies

- **Core**: LangChain, LangGraph, Streamlit, Chroma, HuggingFace Transformers
- **Scraping**: BeautifulSoup, Requests
- **AI/ML**: Sentence Transformers, Scikit-learn, Pandas
- **APIs**: Google Generative AI, Tavily

## Contributing

Feel free to open issues or pull requests for improvements, such as adding more models, enhancing the UI, or integrating notebook features into the main app.

## License

This project is for educational purposes. Ensure compliance with web scraping terms and API usage policies.
