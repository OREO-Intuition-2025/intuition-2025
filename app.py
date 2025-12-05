"""
Streamlit UI for the Change Management Chat Assistant.
Provides an interactive chat interface powered by the RAG workflow.
"""
import streamlit as st
import logging

from src.config import (
    APP_TITLE,
    KNOWLEDGE_BASE_DIR,
    CACHE_DB_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TAVILY_API_KEY,
    WEB_SEARCH_K,
)
from src.document_processor import DocumentProcessor
from src.rag_workflow import RAGWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@st.cache_resource
def initialize_system():
    """
    Initialize the document processor and RAG workflow.
    Uses Streamlit caching to avoid reloading on every interaction.
    
    Returns:
        RAGWorkflow instance ready for queries
    """
    logger.info("Initializing system...")
    
    # Initialize document processor
    doc_processor = DocumentProcessor(
        knowledge_base_dir=KNOWLEDGE_BASE_DIR,
        cache_dir=CACHE_DB_DIR,
        embedding_model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    # Process knowledge base and get retriever
    db, retriever = doc_processor.process_knowledge_base()
    
    # Initialize RAG workflow
    rag_workflow = RAGWorkflow(
        retriever=retriever,
        llm_model=LLM_MODEL,
        tavily_api_key=TAVILY_API_KEY,
        web_search_k=WEB_SEARCH_K,
    )
    
    logger.info("System initialized successfully")
    return rag_workflow


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🔄",
        layout="wide",
    )
    
    # App title and description
    st.title(f"🔄 {APP_TITLE}")
    st.markdown("""
    Welcome to **Metamorphosis** - your AI-powered assistant for change management guidance.
    
    Ask questions about:
    - **Change Management Models**: Lewin's Model, Kotter's 8-Step, McKinsey 7S, ADKAR
    - **Organizational Change Strategies**
    - **Resistance Management**
    - **Transformation Best Practices**
    """)
    
    # Initialize system (cached)
    try:
        with st.spinner("Loading knowledge base..."):
            rag_workflow = initialize_system()
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        return
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How can I help with your change management needs?"):
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Pass chat history (excluding current message) to maintain context
                    history_for_context = st.session_state.messages[:-1] if len(st.session_state.messages) > 0 else []
                    
                    # Stream workflow execution with chat history
                    stream = rag_workflow.query(prompt, chat_history=history_for_context)
                    full_response = ""
                    
                    # Extract generation from workflow output
                    for chunk in stream:
                        if "generate" in chunk:
                            response = chunk["generate"]["generation"]
                            if response and response != full_response:
                                full_response = response
                    
                    # Display response
                    if full_response:
                        st.markdown(full_response)
                    else:
                        full_response = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                        st.warning(full_response)
                    
                except Exception as e:
                    full_response = f"An error occurred: {str(e)}"
                    st.error(full_response)
                    logger.error(f"Query error: {str(e)}", exc_info=True)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This assistant uses:
        - **RAG** (Retrieval-Augmented Generation)
        - **Vector Search** for knowledge retrieval
        - **LLM** (Gemini 2.0) for generation
        - **Quality Checks** for accurate responses
        """)
        
        st.header("Features")
        st.markdown("""
        ✅ Grounded answers from curated sources  
        ✅ Web search fallback for current info  
        ✅ Hallucination detection  
        ✅ Answer relevance checking  
        ✅ Conversation memory (last 3 turns)  
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
