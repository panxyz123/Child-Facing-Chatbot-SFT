"""
LangGraph Agent Pipeline with RAG and TinyLlama

This agent integrates:
- LanceDB for vector retrieval (RAG)
- TinyLlama as the generation model (served by vLLM, or Ollama as fallback)
- LangGraph for orchestration and state management

The agent flow:
1. User query → Retrieve relevant documents from LanceDB
2. Build context from retrieved documents
3. Generate answer using TinyLlama with the context
4. Return final response with citations
"""

from typing import Any, Optional, List
from dataclasses import dataclass, field
import json
import os

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from typing_extensions import TypedDict

# LLM integration
from langchain_community.llms import Ollama  # For local TinyLlama
from langchain_openai import ChatOpenAI  # vLLM exposes an OpenAI-compatible API
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# RAG: Vector retrieval
import lancedb
from sentence_transformers import SentenceTransformer

# ======================== CONFIG ========================

# LanceDB storage path (local or S3)
LANCE_STORAGE_PATH = os.environ.get("LANCE_STORAGE_PATH", "./lance_db")
LANCE_TABLE_NAME = os.environ.get("LANCE_TABLE_NAME", "wiki_science_v2_fixed")

# Embedding model for retrieval
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# TinyLlama config
TINYLLAMA_MODEL = "tinyllama:latest"  # Ollama model name
TINYLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server URL

# vLLM config (start with: vllm serve <model> --dtype half --served-model-name tinyllama)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "vllm")  # "vllm" or "ollama"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "tinyllama")

# Retrieval config
TOP_K_RETRIEVAL = 3  # Number of documents to retrieve

# ======================== STATE DEFINITION ========================

class AgentState(TypedDict):
    """Agent execution state passed between graph nodes."""
    
    query: str  # User's question
    retrieved_docs: List[dict]  # Retrieved context documents
    context: str  # Formatted context for LLM
    answer: str  # Final generated answer
    citations: List[str]  # Sources used in answer
    error: Optional[str]  # Error message if any


# ======================== RAG RETRIEVAL ========================

class RAGRetriever:
    """Vector retrieval from LanceDB."""
    
    def __init__(self, storage_path: str = LANCE_STORAGE_PATH,
                 table_name: str = LANCE_TABLE_NAME,
                 embedding_model_name: str = EMBEDDING_MODEL_NAME):
        """Initialize retriever with LanceDB connection and embedding model."""
        self.storage_path = storage_path
        self.table_name = table_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Connect to LanceDB
        try:
            self.db = lancedb.connect(storage_path)
            self.table = self.db.open_table(table_name)
            print(f"✓ Connected to LanceDB table: {table_name}")
        except Exception as e:
            print(f"⚠ Failed to connect to LanceDB: {e}")
            self.table = None
    
    def retrieve(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[dict]:
        """Retrieve top-k documents similar to the query."""
        if self.table is None:
            return []
        
        try:
            # Encode query to vector
            query_vector = self.embedding_model.encode([query])[0].tolist()
            
            # Search in LanceDB (using vector similarity)
            results = self.table.search(query_vector).metric("cosine").limit(k).to_list()
            
            docs = []
            for r in results:
                docs.append({
                    "id": r.get("id"),
                    "title": r.get("topic") or r.get("title", "Unknown"),
                    "text": r.get("search_text") or r.get("content", ""),
                    "category": r.get("category", ""),
                    "url": r.get("url", "")
                })
            
            return docs
        except Exception as e:
            print(f"⚠ Retrieval error: {e}")
            return []


# ======================== LLM GENERATION ========================

class TinyLlamaGenerator:
    """TinyLlama-based response generator."""
    
    def __init__(self, model: str = TINYLLAMA_MODEL,
                 base_url: str = TINYLLAMA_BASE_URL,
                 backend: str = LLM_BACKEND):
        """Initialize TinyLlama via vLLM (default) or Ollama."""
        self.model_name = model
        
        try:
            if backend == "vllm":
                self.llm = ChatOpenAI(
                    model=VLLM_MODEL,
                    base_url=VLLM_BASE_URL,
                    api_key="EMPTY",
                    temperature=0,
                    streaming=True,
                    callbacks=[StreamingStdOutCallbackHandler()]
                )
            else:
                self.llm = Ollama(
                    model=model,
                    base_url=base_url,
                    callbacks=[StreamingStdOutCallbackHandler()]
                )
            print(f"✓ Initialized TinyLlama ({backend})")
        except Exception as e:
            print(f"⚠ Failed to initialize TinyLlama: {e}")
            self.llm = None
    
    def generate(self, query: str, context: str) -> str:
        """Generate answer based on query and context."""
        if self.llm is None:
            return "Error: TinyLlama not initialized."
        
        # Construct prompt with context
        prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""Based on the following context, answer the question concisely.

Context:
{context}

Question: {query}

Answer:"""
        )
        
        try:
            prompt = prompt_template.format(context=context, query=query)
            answer = self.llm.invoke(prompt)
            answer = getattr(answer, "content", answer)  # ChatOpenAI returns an AIMessage
            return answer.strip()
        except Exception as e:
            return f"Error generating answer: {e}"


# ======================== GRAPH NODES ========================

def node_retrieve(state: AgentState, retriever: RAGRetriever) -> dict:
    """Node 1: Retrieve relevant documents from RAG."""
    query = state["query"]
    print(f"\n🔍 Retrieving documents for: '{query}'")
    
    docs = retriever.retrieve(query, k=TOP_K_RETRIEVAL)
    
    return {
        "retrieved_docs": docs,
    }


def node_format_context(state: AgentState) -> dict:
    """Node 2: Format retrieved documents into context string."""
    docs = state["retrieved_docs"]
    
    if not docs:
        context = "No relevant documents found in knowledge base."
    else:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[Source {i}]\n"
                f"Title: {doc['title']}\n"
                f"Category: {doc['category']}\n"
                f"Content: {doc['text'][:500]}...\n"
            )
        context = "\n".join(context_parts)
    
    return {
        "context": context,
    }


def node_generate(state: AgentState, generator: TinyLlamaGenerator) -> dict:
    """Node 3: Generate answer using TinyLlama."""
    query = state["query"]
    context = state["context"]
    
    print(f"\n💡 Generating answer with TinyLlama...")
    
    answer = generator.generate(query, context)
    
    # Extract citations from retrieved docs
    citations = [
        f"{doc['title']} ({doc['url']})" 
        for doc in state["retrieved_docs"]
    ]
    
    return {
        "answer": answer,
        "citations": citations,
    }


def node_format_output(state: AgentState) -> dict:
    """Node 4: Format final output with citations."""
    answer = state["answer"]
    citations = state["citations"]
    
    output = f"\n{'='*60}\nFinal Answer:\n{answer}\n"
    
    if citations:
        output += f"\nSources:\n"
        for cite in citations:
            output += f"  • {cite}\n"
    
    output += f"{'='*60}\n"
    
    return {"answer": output}


# ======================== AGENT PIPELINE ========================

class RAGAgent:
    """LangGraph-based RAG Agent with TinyLlama."""
    
    def __init__(self, lance_storage_path: str = LANCE_STORAGE_PATH,
                 lance_table_name: str = LANCE_TABLE_NAME):
        """Initialize agent with RAG retriever and TinyLlama generator."""
        
        # Initialize components
        self.retriever = RAGRetriever(lance_storage_path, lance_table_name)
        self.generator = TinyLlamaGenerator()
        
        # Build LangGraph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph workflow."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node(
            "retrieve",
            lambda state: node_retrieve(state, self.retriever)
        )
        workflow.add_node(
            "format_context",
            node_format_context
        )
        workflow.add_node(
            "generate",
            lambda state: node_generate(state, self.generator)
        )
        workflow.add_node(
            "format_output",
            node_format_output
        )
        
        # Define edges (workflow)
        workflow.add_edge("retrieve", "format_context")
        workflow.add_edge("format_context", "generate")
        workflow.add_edge("generate", "format_output")
        workflow.add_edge("format_output", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Compile graph
        return workflow.compile()
    
    def run(self, query: str) -> str:
        """Execute the agent pipeline for a given query."""
        
        initial_state = {
            "query": query,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "citations": [],
            "error": None,
        }
        
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state["answer"]
        except Exception as e:
            print(f"❌ Agent execution error: {e}")
            return f"Error: {e}"


# ======================== MAIN / EXAMPLE ========================

if __name__ == "__main__":
    # Example usage
    print("Initializing RAG Agent with TinyLlama...")
    agent = RAGAgent()
    
    # Example queries
    queries = [
        "What is artificial intelligence?",
        "Tell me about quantum mechanics",
    ]
    
    for query in queries:
        response = agent.run(query)
        print(response)
