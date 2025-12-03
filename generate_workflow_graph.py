"""
Generate a visual diagram of the RAG workflow graph.
This creates a PNG image showing the complete flow with all nodes, edges, and decision points.
"""
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List
import os

# Set environment variables to avoid API key requirements for graph generation
os.environ.setdefault("GOOGLE_API_KEY", "dummy_key_for_graph_generation")
os.environ.setdefault("TAVILY_API_KEY", "dummy_key_for_graph_generation")

# Import after setting env vars
from src.config import (
    KNOWLEDGE_BASE_DIR,
    CACHE_DB_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TAVILY_API_KEY,
    WEB_SEARCH_K,
)


class GraphState(TypedDict):
    """State that flows through the workflow."""
    question: str
    generation: str
    documents: List[str]
    source: str


def route_question(state):
    """Route to vectorstore or web search."""
    return "vectorstore"  # Dummy implementation


def route_after_transform(state):
    """Route back to original source after transform."""
    return state.get("source", "vectorstore")


def decide_to_generate(state):
    """Decide whether to generate or transform query."""
    return "generate"  # Dummy implementation


def grade_generation(state):
    """Grade the generated answer."""
    return "useful"  # Dummy implementation


def retrieve(state):
    """Retrieve documents."""
    return state


def grade_documents(state):
    """Grade document relevance."""
    return state


def generate(state):
    """Generate answer."""
    return state


def transform_query(state):
    """Transform the query."""
    return state


def web_search(state):
    """Perform web search."""
    return state


# Build the workflow graph
workflow = StateGraph(GraphState)

# Add all nodes
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Set conditional entry point
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)

# Add edges
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")

# Add conditional edges
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_conditional_edges(
    "transform_query",
    route_after_transform,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)

workflow.add_conditional_edges(
    "generate",
    grade_generation,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile the graph
app = workflow.compile()

# Generate the graph visualization
try:
    # Get the graph as a Mermaid diagram first (for debugging)
    print("Generating workflow graph...")
    
    # Generate PNG
    png_data = app.get_graph().draw_mermaid_png()
    
    # Save to file
    output_path = "workflow_graph.png"
    with open(output_path, "wb") as f:
        f.write(png_data)
    
    print(f"✅ Graph saved to {output_path}")
    print("\nGraph shows the complete RAG workflow:")
    print("- Entry: route_question (vectorstore vs web_search)")
    print("- Main path: retrieve → grade_documents → generate")
    print("- Loops: transform_query → retrieve (retry with better question)")
    print("- Quality checks: generate → grade → END or retry")
    
except Exception as e:
    print(f"❌ Error generating graph: {e}")
    print("\nTrying alternative method with graphviz...")
    
    try:
        # Alternative: ASCII representation
        from langgraph.graph import Graph
        print("\nWorkflow structure:")
        print(app.get_graph().draw_ascii())
    except Exception as e2:
        print(f"ASCII method also failed: {e2}")
