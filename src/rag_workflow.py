"""
RAG (Retrieval-Augmented Generation) workflow using LangGraph.
Implements the complete question-answering pipeline with quality checks.
"""
from typing import List
from typing_extensions import TypedDict
import json
import re

from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from .prompts import (
    question_router_prompt,
    document_grader_prompt,
    hallucination_grader_prompt,
    answer_grader_prompt,
    question_rewriter_prompt,
)
import logging

logger = logging.getLogger(__name__)


class SafeJsonOutputParser(BaseOutputParser):
    """Output parser that handles both single and double quotes in JSON."""
    
    def parse(self, text: str) -> dict:
        """
        Parse JSON text, handling common formatting issues.
        
        Args:
            text: String that should contain JSON
            
        Returns:
            Parsed dictionary
        """
        # Remove markdown formatting if present
        text = re.sub(r'^```json\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
        text = text.strip()
        
        # Try standard JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try replacing single quotes with double quotes
            try:
                fixed_text = text.replace("'", '"')
                return json.loads(fixed_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {text}")
                logger.error(f"Error: {e}")
                raise ValueError(f"Could not parse JSON: {text}")


class GraphState(TypedDict):
    """
    Represents the state of the RAG workflow graph.
    
    Attributes:
        question: User's question
        generation: LLM-generated answer
        documents: Retrieved or searched documents
        source: Source of documents ('vectorstore' or 'web_search')
        retry_count: Number of retries attempted
    """
    question: str
    generation: str
    documents: List[str]
    source: str
    retry_count: int


class RAGWorkflow:
    """Implements the RAG workflow with quality checks using LangGraph."""
    
    def __init__(self, retriever, llm_model, tavily_api_key, web_search_k=3):
        """
        Initialize the RAG workflow.
        
        Args:
            retriever: Document retriever for vectorstore queries
            llm_model: Name of the LLM model to use
            tavily_api_key: API key for Tavily web search
            web_search_k: Number of web search results to retrieve
        """
        self.retriever = retriever
        self.llm_model = llm_model
        self.web_search_k = web_search_k
        
        # Initialize LLM for JSON outputs (routing, grading)
        self.llm_json = ChatGoogleGenerativeAI(model=llm_model, format="json")
        
        # Initialize LLM for text outputs (generation)
        self.llm_text = ChatGoogleGenerativeAI(model=llm_model)
        
        # Initialize web search tool
        self.web_search_tool = TavilySearchResults(k=web_search_k, tavily_api_key=tavily_api_key)
        
        # Build chains
        self._build_chains()
        
        # Build workflow graph
        self.app = self._build_graph()
        
    def _build_chains(self):
        """Build all LangChain chains for the workflow."""
        logger.info("Building RAG chains...")
        # RAG Chains are of the format: prompt | llm | output_parser
        # They are used to process inputs and produce outputs in various nodes of the workflow.
        
        # Create safe JSON parser instance
        safe_parser = SafeJsonOutputParser()
        
        # Question router chain - with safe JSON parsing
        self.question_router = question_router_prompt | self.llm_json | safe_parser
        
        # Document grader chain - with safe JSON parsing
        self.grader = document_grader_prompt | self.llm_json | safe_parser
        
        # RAG answer generation chain (create local prompt)
        rag_prompt = PromptTemplate.from_template(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise and helpful.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        self.rag_chain = rag_prompt | self.llm_text | StrOutputParser()
        
        # Hallucination grader chain - with safe JSON parsing
        self.hallucination_grader = hallucination_grader_prompt | self.llm_json | safe_parser
        
        # Answer grader chain - with safe JSON parsing
        self.answer_grader = answer_grader_prompt | self.llm_json | safe_parser
        
        # Question rewriter chain
        self.question_rewriter = question_rewriter_prompt | self.llm_text | StrOutputParser()
        
        logger.info("RAG chains built successfully")
    
    # Node functions
    def retrieve(self, state):
        """
        Retrieve documents from vectorstore.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with retrieved documents
        """
        logger.info("---RETRIEVE---")
        question = state["question"]
        
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question, "source": "vectorstore"}
    
    def generate(self, state):
        """
        Generate answer using retrieved documents.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with generated answer
        """
        logger.info("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state):
        """
        Filter documents based on relevance to question.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with filtered documents
        """
        logger.info("---CHECK DOCUMENT RELEVANCE---")
        question = state["question"]
        documents = state["documents"]
        source = state.get("source", "vectorstore")
        
        filtered_docs = []
        for d in documents:
            score = self.grader.invoke({"question": question, "document": d.page_content})
            grade = score["score"]
            if grade == "yes":
                logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
                
        return {"documents": filtered_docs, "question": question, "source": source}
    
    def transform_query(self, state):
        """
        Transform query to improve retrieval.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with transformed question
        """
        logger.info("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        source = state.get("source", "vectorstore")
        retry_count = state.get("retry_count", 0)
        
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question, "source": source, "retry_count": retry_count + 1}
    
    def web_search(self, state):
        """
        Perform web search for current question.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with web search results
        """
        logger.info("---WEB SEARCH---")
        question = state["question"]
        
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        
        return {"documents": web_results, "question": question, "source": "web_search"}
    
    # Edge functions (routing logic)
    def route_question(self, state):
        """
        Route question to vectorstore or web search.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node to execute
        """
        logger.info("---ROUTE QUESTION---")
        question = state["question"]
        logger.info(f"Question: {question}")
        
        source = self.question_router.invoke({"question": question})
        logger.info(f"Routing to: {source['datasource']}")
        
        if source["datasource"] == "web_search":
            return "web_search"
        elif source["datasource"] == "vectorstore":
            return "vectorstore"
    
    def decide_to_generate(self, state):
        """
        Decide whether to generate answer or transform query.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node to execute
        """
        logger.info("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        
        if not filtered_documents:
            logger.info("---DECISION: TRANSFORM QUERY---")
            return "transform_query"
        else:
            logger.info("---DECISION: GENERATE---")
            return "generate"
    
    def route_after_transform(self, state):
        """
        Route back to appropriate source after transforming query.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node to execute
        """
        source = state.get("source", "vectorstore")
        logger.info(f"---ROUTE AFTER TRANSFORM: {source}---")
        return source
    
    def grade_generation_v_documents_and_question(self, state):
        """
        Check if generation is grounded and useful.
        
        Args:
            state: Current graph state
            
        Returns:
            Next node to execute or END
        """
        logger.info("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        retry_count = state.get("retry_count", 0)
        
        # If we've retried too many times, just return the answer
        if retry_count >= 3:
            logger.warning(f"---MAX RETRIES REACHED ({retry_count}), RETURNING ANSWER---")
            return "useful"
        
        # Check if answer is grounded in documents
        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]
        
        if grade == "yes":
            logger.info("---DECISION: GENERATION IS GROUNDED---")
            
            # Check if answer is useful
            logger.info("---GRADE GENERATION vs QUESTION---")
            score = self.answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            
            if grade == "yes":
                logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            logger.info("---DECISION: GENERATION NOT GROUNDED, RETRY---")
            return "not supported"
    
    def _build_graph(self):
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled workflow application
        """
        logger.info("Building workflow graph...")
        
        workflow = StateGraph(GraphState)
        
        # Define nodes
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        
        # Build graph structure
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        
        workflow.add_conditional_edges(
            "transform_query",
            self.route_after_transform,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "transform_query",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        
        logger.info("Workflow graph built successfully")
        
        # Compile with increased recursion limit
        return workflow.compile({"recursion_limit": 50})
    
    def query(self, question):
        """
        Process a user question through the workflow.
        
        Args:
            question: User's question string
            
        Returns:
            Generator yielding workflow state updates
        """
        return self.app.stream({"question": question})
