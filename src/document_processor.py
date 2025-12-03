"""
Document loading and processing module.
Handles loading, splitting, and embedding of knowledge base documents.
"""
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading, splitting, and embedding operations."""
    
    def __init__(self, knowledge_base_dir, cache_dir, embedding_model_name, chunk_size, chunk_overlap):
        """
        Initialize the document processor.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base documents
            cache_dir: Directory for caching embeddings
            embedding_model_name: Name of the HuggingFace embedding model
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between consecutive chunks
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.cache_dir = cache_dir
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.embeddings_model = None
        self.text_splitter = None
        
    def initialize_components(self):
        """Initialize text splitter and embedding models."""
        logger.info("Initializing document processor components...")
        
        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Initialize embedding model with cache_folder for persistence
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            cache_folder=self.cache_dir
        )
        
        logger.info("Document processor initialized successfully")
        
    def load_documents(self):
        """
        Load documents from the knowledge base directory.
        
        Returns:
            List of loaded Document objects
        """
        logger.info(f"Loading documents from {self.knowledge_base_dir}...")
        
        loader = DirectoryLoader(
            self.knowledge_base_dir,
            glob="**/*.txt", # Purpose: Load all .txt files
            show_progress=True,
            use_multithreading=True,
        )
        docs = loader.load()
        
        logger.info(f"Loaded {len(docs)} documents")
        return docs
    
    def split_documents(self, docs):
        """
        Split documents into smaller chunks.
        
        Args:
            docs: List of Document objects to split
            
        Returns:
            List of split Document chunks
        """
        logger.info("Splitting documents into chunks...")
        
        if not self.text_splitter:
            self.initialize_components()
            
        texts = self.text_splitter.split_documents(docs)
        
        logger.info(f"Created {len(texts)} document chunks")
        return texts
    
    def create_vector_store(self, texts):
        """
        Create a Chroma vector store from document chunks.
        
        Args:
            texts: List of Document chunks to embed and store
            
        Returns:
            Chroma vector store instance
        """
        logger.info("Creating vector store with embeddings...")
        
        if not self.embeddings_model:
            self.initialize_components()
            
        db = Chroma.from_documents(texts, self.embeddings_model)
        
        logger.info("Vector store created successfully")
        return db
    
    def process_knowledge_base(self):
        """
        Complete pipeline: load, split, and create vector store.
        
        Returns:
            Tuple of (vector_store, retriever)
        """
        logger.info("Starting knowledge base processing pipeline...")
        
        # Initialize all components
        self.initialize_components()
        
        # Load and process documents
        docs = self.load_documents()
        texts = self.split_documents(docs)
        db = self.create_vector_store(texts)
        
        # Create retriever for querying
        retriever = db.as_retriever()
        
        logger.info("Knowledge base processing complete")
        return db, retriever
