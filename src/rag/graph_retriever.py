"""
src/rag/graph_retriever.py — Graph-augmented retrieval using the knowledge graph.

Combines FAISS vector similarity with graph traversal for richer context:
1. Find relevant entities via text matching
2. Traverse 1-2 hops in the graph to find connected chunks
3. Merge with FAISS results, deduplicate, and re-rank

Usage:
    from src.rag.graph_retriever import hybrid_retrieve
    chunks = hybrid_retrieve(query, k=5)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import KNOWLEDGE_GRAPH_PATH

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _NX_OK = True
except ImportError:
    _NX_OK = False


def graph_query(
    query: str,
    graph: Any = None,
    k: int = 3,
) -> list[dict]:
    """
    Query the knowledge graph for relevant chunks.

    Strategy:
    1. Tokenize query into keywords
    2. Find entity nodes matching keywords
    3. Traverse 1-2 hops to find connected chunk nodes
    4. Score chunks by number of matching entity connections
    5. Return top-k chunks
    """
    if not _NX_OK:
        return []

    if graph is None:
        try:
            from src.rag.graph_builder import load_graph
            graph = load_graph()
        except (FileNotFoundError, ImportError):
            return []

    query_lower = query.lower()
    query_tokens = set(query_lower.split())

    # Find matching entity nodes
    matching_entities = []
    for node, data in graph.nodes(data=True):
        if data.get("node_type") != "entity":
            continue
        node_lower = str(node).lower()
        # Score: how many query tokens appear in the entity name
        score = sum(1 for t in query_tokens if t in node_lower)
        if score > 0:
            matching_entities.append((node, score))

    matching_entities.sort(key=lambda x: x[1], reverse=True)
    matching_entities = matching_entities[:15]

    if not matching_entities:
        return []

    # Traverse graph: entity → connected chunks (1-2 hops)
    chunk_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for entity_name, entity_score in matching_entities:
        # Direct neighbors (1 hop)
        for neighbor in graph.neighbors(entity_name):
            ndata = graph.nodes.get(neighbor, {})
            if ndata.get("node_type") == "chunk":
                chunk_scores[neighbor] = chunk_scores.get(neighbor, 0) + entity_score * 1.0
                chunk_data[neighbor] = ndata

        # 2-hop: entity → entity → chunk
        for mid_node in graph.neighbors(entity_name):
            mid_data = graph.nodes.get(mid_node, {})
            if mid_data.get("node_type") != "entity":
                continue
            for neighbor in graph.neighbors(mid_node):
                ndata = graph.nodes.get(neighbor, {})
                if ndata.get("node_type") == "chunk":
                    chunk_scores[neighbor] = chunk_scores.get(neighbor, 0) + entity_score * 0.5
                    chunk_data[neighbor] = ndata

    # Sort by score and return top-k
    ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for chunk_id, score in ranked:
        data = chunk_data.get(chunk_id, {})
        results.append({
            "chunk_id": chunk_id,
            "source": data.get("source", ""),
            "text": data.get("text", ""),
            "score": float(score),
            "retrieval_method": "graph",
        })

    logger.debug("Graph query returned %d chunks for: %s", len(results), query[:60])
    return results


def get_entity_neighbors(
    entity_name: str,
    graph: Any = None,
    max_depth: int = 2,
) -> dict[str, Any]:
    """
    Get all neighbors of an entity within max_depth hops.
    Useful for UI visualization of the knowledge graph around a concept.
    """
    if not _NX_OK:
        return {"entity": entity_name, "neighbors": []}

    if graph is None:
        try:
            from src.rag.graph_builder import load_graph
            graph = load_graph()
        except (FileNotFoundError, ImportError):
            return {"entity": entity_name, "neighbors": []}

    # Find the entity node (case-insensitive)
    target = None
    for node in graph.nodes():
        if str(node).lower() == entity_name.lower():
            target = node
            break

    if target is None:
        return {"entity": entity_name, "neighbors": []}

    # BFS up to max_depth
    neighbors = []
    visited = {target}
    queue = [(target, 0)]

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue

        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue
            visited.add(neighbor)

            ndata = graph.nodes.get(neighbor, {})
            edge_data = graph.edges.get((current, neighbor), {})

            neighbors.append({
                "name": str(neighbor),
                "type": ndata.get("entity_type", ndata.get("node_type", "unknown")),
                "relationship": edge_data.get("relationship", "connected"),
                "depth": depth + 1,
            })
            queue.append((neighbor, depth + 1))

    return {
        "entity": entity_name,
        "neighbors": neighbors,
        "total": len(neighbors),
    }


def hybrid_retrieve(
    query: str,
    k: int = 5,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
) -> list[dict]:
    """
    Hybrid retrieval combining FAISS vector search + graph traversal.

    1. Run FAISS query → vector_results
    2. Run graph query → graph_results
    3. Merge, deduplicate by chunk_id, re-rank by weighted score
    """
    from src.rag.retriever import rag_query

    # Vector search (existing FAISS)
    vector_results = rag_query(query, k=k)
    for r in vector_results:
        r["retrieval_method"] = "vector"

    # Graph search
    graph_results = graph_query(query, k=k)

    # Merge and score
    all_chunks: dict[str, dict] = {}

    for r in vector_results:
        cid = r.get("chunk_id", id(r))
        if cid not in all_chunks:
            all_chunks[cid] = r.copy()
            all_chunks[cid]["combined_score"] = r.get("score", 0.5) * vector_weight
        else:
            all_chunks[cid]["combined_score"] += r.get("score", 0.5) * vector_weight

    for r in graph_results:
        cid = r.get("chunk_id", id(r))
        if cid not in all_chunks:
            all_chunks[cid] = r.copy()
            all_chunks[cid]["combined_score"] = r.get("score", 0.5) * graph_weight
        else:
            all_chunks[cid]["combined_score"] += r.get("score", 0.5) * graph_weight
            all_chunks[cid]["retrieval_method"] = "hybrid"

    # Sort by combined score
    ranked = sorted(all_chunks.values(), key=lambda x: x.get("combined_score", 0), reverse=True)

    return ranked[:k]
