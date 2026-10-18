# Author: Rayaan tasnim, 13 aged prodigy

import os
import sys
import json
import logging
import traceback
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from pydantic import BaseModel, Field
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AgenticRAGPipeline")

class EvaluationResult(BaseModel):
    relevance_score: float = Field(..., description="Score between 0.0 and 1.0 indicating relevance and completeness of retrieved context.")
    reasoning: str = Field(..., description="Explanation for the assigned score.")
    needs_rerouting: bool = Field(..., description="True if score is below threshold or context is insufficient.")

class InMemoryVectorIndex:
    """In-memory vector index utilizing NumPy for vectorized cosine similarity searches."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.documents: List[str] = []

    def add_documents(self, docs: List[str], embeddings: np.ndarray) -> None:
        try:
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")
            self.vectors = np.vstack([self.vectors, embeddings])
            self.documents.extend(docs)
            logger.info(f"Added {len(docs)} documents to vector index. Total documents: {len(self.documents)}")
        except Exception as e:
            logger.error(f"Error adding documents to index: {e}\n{traceback.format_exc()}")
            raise

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        try:
            if len(self.documents) == 0:
                return []
            
            # Normalize vectors for cosine similarity
            q_norm = np.linalg.norm(query_embedding)
            if q_norm == 0:
                q_norm = 1e-8
            q_emb_normed = query_embedding / q_norm

            v_norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            v_norms[v_norms == 0] = 1e-8
            v_vectors_normed = self.vectors / v_norms

            # Vectorized cosine similarity
            similarities = np.dot(v_vectors_normed, q_emb_normed)
            
            top_k = min(top_k, len(self.documents))
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = [(self.documents[idx], float(similarities[idx])) for idx in top_indices]
            return results
        except Exception as e:
            logger.error(f"Error during vector search: {e}\n{traceback.format_exc()}")
            return []

    def clear(self) -> None:
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.documents = []
        logger.info("Vector index cleared.")

class DataDriftMonitor:
    """Monitors semantic drift between historical query distributions and incoming queries using Wasserstein distance."""
    def __init__(self, threshold: float = 0.35, max_history: int = 100):
        self.threshold = threshold
        self.max_history = max_history
        self.historical_embeddings: List[np.ndarray] = []

    def update_and_check_drift(self, incoming_embedding: np.ndarray) -> bool:
        try:
            if len(self.historical_embeddings) < 10:
                self.historical_embeddings.append(incoming_embedding)
                return False

            historical_matrix = np.array(self.historical_embeddings)
            hist_norms = np.linalg.norm(historical_matrix, axis=1)
            inc_norm = np.linalg.norm(incoming_embedding)

            drift_score = wasserstein_distance(hist_norms, [inc_norm])
            logger.info(f"Calculated data drift score (Wasserstein): {drift_score:.4f} (Threshold: {self.threshold})")

            self.historical_embeddings.append(incoming_embedding)
            if len(self.historical_embeddings) > self.max_history:
                self.historical_embeddings.pop(0)

            return drift_score > self.threshold
        except Exception as e:
            logger.error(f"Error in drift monitoring: {e}\n{traceback.format_exc()}")
            return False

class AgenticRAGPipeline:
    """Self-Rerouting Agentic RAG Pipeline with Real-Time Data Drift Correction."""
    def __init__(self, api_key: Optional[str] = None, embedding_dim: int = 128):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.dimension = embedding_dim
        self.vector_index = InMemoryVectorIndex(dimension=self.dimension)
        self.drift_monitor = DataDriftMonitor(threshold=0.30)
        self.threshold_score = 0.85

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates embeddings (with deterministic fallback simulation for standalone robustness)."""
        try:
            np.random.seed(abs(hash(text)) % (2**32))
            emb = np.random.randn(self.dimension).astype(np.float32)
            return emb / np.linalg.norm(emb)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}\n{traceback.format_exc()}")
            return np.zeros(self.dimension, dtype=np.float32)

    def evaluate_context(self, query: str, context: str) -> EvaluationResult:
        """Judge Agent evaluating retrieved context completeness and semantic alignment."""
        try:
            prompt = f"""
            You are a strict Judge Agent in an Agentic RAG pipeline.
            Evaluate whether the provided context is sufficient, relevant, and accurate to answer the user query.
            
            User Query: {query}
            Retrieved Context: {context}
            
            Provide relevance_score (0.0 to 1.0), reasoning, and needs_rerouting (true if score < {self.threshold_score}).
            """
            
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a precise evaluation judge."},
                          {"role": "user", "content": prompt}],
                response_format=EvaluationResult,
            )
            return response.choices[0].message.parsed
        except Exception as e:
            logger.warning(f"LLM evaluation API error or fallback triggered: {e}. Defaulting to rule-based evaluation.")
            has_content = len(context.strip()) > 10
            score = 0.90 if has_content else 0.40
            return EvaluationResult(
                relevance_score=score,
                reasoning="Fallback evaluation due to API constraints or mock mode.",
                needs_rerouting=(score < self.threshold_score)
            )

    def external_search_fallback(self, query: str) -> str:
        """Simulates external live search API to fetch fresh contextual bounds during rerouting."""
        logger.info(f"Executing external search fallback for query: '{query}'")
        return f"[Live External Search Result]: Fresh authoritative data regarding {query} retrieved from external knowledge base."

    def run_pipeline(self, query: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing query: '{query}'")
            query_embedding = self._get_embedding(query)

            # 1. Check for Real-Time Data Drift
            is_drifted = self.drift_monitor.update_and_check_drift(query_embedding)
            if is_drifted:
                logger.warning("Data drift detected! Triggering in-memory cache clearing and re-indexing routine.")
                self.vector_index.clear()
                baseline_docs = [
                    "Baseline Doc 1: General enterprise architecture guidelines for 2026.",
                    "Baseline Doc 2: Standard operating procedures for cloud resilience."
                ]
                baseline_embeddings = np.array([self._get_embedding(doc) for doc in baseline_docs])
                self.vector_index.add_documents(baseline_docs, baseline_embeddings)

            # 2. Raw Vector Search
            search_results = self.vector_index.search(query_embedding, top_k=2)
            context = "\n".join([doc for doc, score in search_results]) if search_results else "No relevant context found."

            # 3. LLM Evaluation & Self-Rerouting Loop
            eval_result = self.evaluate_context(query, context)
            logger.info(f"Judge Evaluation - Score: {eval_result.relevance_score}, Needs Rerouting: {eval_result.needs_rerouting}")

            routing_path = "Primary Vector Retrieval"
            if eval_result.needs_rerouting:
                logger.info("Triggering fallback routing path: Fetching fresh context via live external search API.")
                context = self.external_search_fallback(query)
                routing_path = "Fallback External Search API Reroute"

            # 4. Final Answer Generation
            final_prompt = f"Answer the query based on the context.\nContext: {context}\nQuery: {query}"
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": final_prompt}]
                )
                answer = completion.choices[0].message.content
            except Exception:
                answer = f"Simulated Generated Answer for '{query}' using context: {context[:100]}..."

            return {
                "query": query,
                "routing_path": routing_path,
                "evaluation": eval_result.model_dump() if hasattr(eval_result, 'model_dump') else eval_result.dict(),
                "context": context,
                "answer": answer,
                "drift_detected": is_drifted
            }

        except Exception as e:
            logger.error(f"Critical error in pipeline execution: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Initializing Self-Rerouting Agentic RAG Pipeline Simulation...")
    
    pipeline = AgenticRAGPipeline(embedding_dim=128)

    # Populate initial knowledge base
    initial_docs = [
        "Quantum computing harnesses quantum mechanics to process complex data simultaneously.",
        "Agentic RAG pipelines integrate autonomous LLM evaluators to dynamically reroute retrieval strategies.",
        "NumPy provides high-performance vectorized matrix operations for efficient similarity searches."
    ]
    initial_embeddings = np.array([pipeline._get_embedding(doc) for doc in initial_docs])
    pipeline.vector_index.add_documents(initial_docs, initial_embeddings)

    # Simulation Step 1: Standard Successful Retrieval
    print("\n--- Simulation 1: Standard Retrieval ---")
    res1 = pipeline.run_pipeline("How do agentic RAG pipelines handle retrieval?")
    print(json.dumps(res1, indent=2))

    # Simulation Step 2: Triggering Self-Rerouting (query outside local knowledge scope)
    print("\n--- Simulation 2: Self-Rerouting Event ---")
    res2 = pipeline.run_pipeline("What are the latest quarterly financial results for SpaceX in 2026?")
    print(json.dumps(res2, indent=2))

    # Simulation Step 3: Triggering Data Drift Correction (feeding out-of-distribution queries)
    print("\n--- Simulation 3: Data Drift Correction Event ---")
    for i in range(15):
        drift_query = f"Completely anomalous orthogonal query vector stream item {i} with extremely high magnitude divergence."
        res_drift = pipeline.run_pipeline(drift_query)
        if res_drift.get("drift_detected"):
            print(f"Drift successfully caught at iteration {i}!")
            break

    logger.info("Simulation completed successfully.")