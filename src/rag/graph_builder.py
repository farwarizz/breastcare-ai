"""
src/rag/graph_builder.py — Build a medical knowledge graph from chunked documents.

Uses Gemini API to extract entities and relationships from each chunk,
then constructs a NetworkX directed graph for graph-augmented retrieval.

Entity types: Disease, Symptom, Procedure, RiskFactor, BiRadsCategory,
              Treatment, Guideline, ImagingModality, Gene
Relationship types: INDICATES, RECOMMENDS, ASSOCIATED_WITH, TREATS,
                    FOLLOWS_UP_WITH, DETECTS, CONTRAINDICATES

Usage:
    from src.rag.graph_builder import build_knowledge_graph, load_graph
    graph = build_knowledge_graph(chunks)
    graph = load_graph()
"""
from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    KNOWLEDGE_GRAPH_PATH, ENTITY_CACHE_PATH,
    COMMUNITY_CACHE_PATH, KB_GRAPHSTORE_DIR,
    get_api_key, GEMINI_MODEL,
)

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _NX_OK = True
except ImportError:
    _NX_OK = False

# Valid entity and relationship types
ENTITY_TYPES = [
    "Disease", "Symptom", "Procedure", "RiskFactor",
    "BiRadsCategory", "Treatment", "Guideline",
    "ImagingModality", "Gene", "Drug", "AnatomicalStructure",
]

RELATIONSHIP_TYPES = [
    "INDICATES", "RECOMMENDS", "ASSOCIATED_WITH", "TREATS",
    "FOLLOWS_UP_WITH", "DETECTS", "CONTRAINDICATES",
    "RISK_FACTOR_FOR", "SCREENS_FOR", "STAGE_OF",
]

EXTRACTION_PROMPT = """Extract medical entities and relationships from this text.

TEXT:
{text}

Return a JSON object with:
- "entities": list of {{"name": str, "type": str}} where type is one of: """ + ", ".join(ENTITY_TYPES) + """
- "relationships": list of {{"source": str, "target": str, "type": str}} where type is one of: """ + ", ".join(RELATIONSHIP_TYPES) + """

Rules:
- Entity names should be normalized (e.g., "mammography" not "Mammographic screening")
- Only extract entities explicitly mentioned in the text
- Keep entity names concise (1-4 words)
- Return valid JSON only, no markdown formatting

JSON:"""


def _chunk_hash(text: str) -> str:
    """Deterministic hash for cache keying."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _load_entity_cache() -> dict[str, Any]:
    """Load cached entity extractions."""
    if ENTITY_CACHE_PATH.exists():
        with open(ENTITY_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_entity_cache(cache: dict) -> None:
    """Persist entity cache to disk."""
    KB_GRAPHSTORE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ENTITY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def extract_entities_llm(text: str, cache: dict) -> dict[str, Any]:
    """
    Extract entities and relationships from text using Gemini API.
    Results are cached to avoid repeated API calls.
    """
    chunk_id = _chunk_hash(text)
    if chunk_id in cache:
        return cache[chunk_id]

    api_key = get_api_key()
    if not api_key:
        logger.warning("No API key — skipping LLM entity extraction")
        return {"entities": [], "relationships": []}

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        response = client.chat.completions.create(
            model=GEMINI_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text[:1500])}],
        )
        raw = response.choices[0].message.content.strip()

        # Clean markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]

        result = json.loads(raw)
        cache[chunk_id] = result
        return result

    except Exception as exc:
        logger.warning("Entity extraction failed for chunk %s: %s", chunk_id, exc)
        return {"entities": [], "relationships": []}


def extract_entities_rule_based(text: str) -> dict[str, Any]:
    """
    Rule-based entity extraction fallback (no API required).
    Uses keyword matching for common medical terms.
    """
    import re

    entities = []
    relationships = []

    # Pattern-based entity detection
    patterns = {
        "BiRadsCategory": [
            r"BI-RADS\s*(?:category\s*)?\d",
            r"category\s*[0-6]",
        ],
        "ImagingModality": [
            r"\b(?:mammograph(?:y|ic)|ultrasound|MRI|breast\s+MRI|tomosynthesis|DBT|FFDM)\b",
        ],
        "Treatment": [
            r"\b(?:lumpectomy|mastectomy|chemotherapy|radiation|hormonal\s+therapy|"
            r"trastuzumab|tamoxifen|biopsy|sentinel\s+node)\b",
        ],
        "RiskFactor": [
            r"\b(?:BRCA[12]|family\s+history|dense\s+breast|obesity|"
            r"alcohol|HRT|hormone\s+replacement|menarche)\b",
        ],
        "Disease": [
            r"\b(?:breast\s+cancer|DCIS|invasive\s+(?:ductal|lobular)|"
            r"carcinoma|malignant|benign|fibroadenoma)\b",
        ],
        "Procedure": [
            r"\b(?:screening|follow-up|clinical\s+breast\s+exam|"
            r"core\s+needle\s+biopsy|FNA|axillary\s+dissection)\b",
        ],
        "Gene": [
            r"\b(?:BRCA1|BRCA2|TP53|PALB2|ATM|CHEK2|CDH1|PTEN|HER2|ER|PR)\b",
        ],
    }

    seen = set()
    for entity_type, regex_list in patterns.items():
        for pattern in regex_list:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group().strip().lower()
                if name not in seen:
                    seen.add(name)
                    entities.append({"name": name, "type": entity_type})

    return {"entities": entities, "relationships": relationships}


def build_knowledge_graph(
    chunks: list[dict],
    use_llm: bool = True,
) -> Any:
    """
    Build a NetworkX knowledge graph from document chunks.

    1. Extract entities from each chunk (LLM or rule-based)
    2. Create nodes for entities with metadata
    3. Create edges for relationships
    4. Link entities to their source chunks
    5. Run community detection
    """
    if not _NX_OK:
        raise ImportError("networkx required: pip install networkx>=3.0")

    graph = nx.DiGraph()
    cache = _load_entity_cache() if use_llm else {}
    total_entities = 0
    total_relationships = 0

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        source = chunk.get("source", "")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")

        # Add chunk node
        graph.add_node(
            chunk_id,
            node_type="chunk",
            text=text[:200],
            source=source,
        )

        # Extract entities
        if use_llm and get_api_key():
            extraction = extract_entities_llm(text, cache)
        else:
            extraction = extract_entities_rule_based(text)

        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        # Add entity nodes and link to chunk
        for entity in entities:
            name = entity.get("name", "").strip().lower()
            etype = entity.get("type", "Unknown")
            if not name:
                continue

            if not graph.has_node(name):
                graph.add_node(name, node_type="entity", entity_type=etype)
                total_entities += 1

            graph.add_edge(chunk_id, name, relationship="CONTAINS")
            graph.add_edge(name, chunk_id, relationship="MENTIONED_IN")

        # Add relationship edges
        for rel in relationships:
            src = rel.get("source", "").strip().lower()
            tgt = rel.get("target", "").strip().lower()
            rtype = rel.get("type", "ASSOCIATED_WITH")
            if src and tgt:
                # Ensure nodes exist
                for node_name in [src, tgt]:
                    if not graph.has_node(node_name):
                        graph.add_node(node_name, node_type="entity", entity_type="Unknown")
                graph.add_edge(src, tgt, relationship=rtype)
                total_relationships += 1

        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d chunks", i + 1, len(chunks))

    # Save entity cache
    if use_llm and cache:
        _save_entity_cache(cache)

    # Run community detection
    communities = _detect_communities(graph)

    logger.info(
        "Knowledge graph built: %d nodes, %d edges, %d entities, %d relationships, %d communities",
        graph.number_of_nodes(), graph.number_of_edges(),
        total_entities, total_relationships, len(communities),
    )

    return graph


def _detect_communities(graph: Any) -> dict[int, list[str]]:
    """
    Detect communities using greedy modularity on the undirected projection.
    """
    if not _NX_OK:
        return {}

    try:
        undirected = graph.to_undirected()
        communities_gen = nx.community.greedy_modularity_communities(undirected)
        communities = {
            i: list(comm) for i, comm in enumerate(communities_gen)
        }

        # Store community IDs on nodes
        for comm_id, members in communities.items():
            for member in members:
                if graph.has_node(member):
                    graph.nodes[member]["community"] = comm_id

        # Save community summaries
        _save_community_summaries(communities, graph)
        return communities
    except Exception as exc:
        logger.warning("Community detection failed: %s", exc)
        return {}


def _save_community_summaries(communities: dict, graph: Any) -> None:
    """Generate and save community summaries."""
    summaries = {}
    for comm_id, members in communities.items():
        entity_members = [
            m for m in members
            if graph.nodes.get(m, {}).get("node_type") == "entity"
        ]
        summaries[str(comm_id)] = {
            "size": len(members),
            "entities": entity_members[:20],
            "entity_types": list(set(
                graph.nodes[m].get("entity_type", "Unknown")
                for m in entity_members
            )),
        }

    KB_GRAPHSTORE_DIR.mkdir(parents=True, exist_ok=True)
    with open(COMMUNITY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def save_graph(graph: Any, path: Path | None = None) -> None:
    """Save knowledge graph to GraphML format."""
    if not _NX_OK:
        return
    path = path or KNOWLEDGE_GRAPH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, str(path))
    logger.info("Knowledge graph saved: %s (%d nodes, %d edges)",
                path, graph.number_of_nodes(), graph.number_of_edges())


def load_graph(path: Path | None = None) -> Any:
    """Load knowledge graph from GraphML format."""
    if not _NX_OK:
        raise ImportError("networkx required")
    path = path or KNOWLEDGE_GRAPH_PATH
    if not path.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {path}")
    graph = nx.read_graphml(str(path))
    logger.info("Knowledge graph loaded: %s (%d nodes, %d edges)",
                path, graph.number_of_nodes(), graph.number_of_edges())
    return graph


def get_graph_stats(graph: Any = None) -> dict[str, Any]:
    """Get statistics about the knowledge graph."""
    if graph is None:
        try:
            graph = load_graph()
        except (FileNotFoundError, ImportError):
            return {"available": False}

    entity_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get("node_type") == "entity"
    ]
    chunk_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get("node_type") == "chunk"
    ]
    entity_types = {}
    for n in entity_nodes:
        etype = graph.nodes[n].get("entity_type", "Unknown")
        entity_types[etype] = entity_types.get(etype, 0) + 1

    return {
        "available": True,
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "entity_count": len(entity_nodes),
        "chunk_count": len(chunk_nodes),
        "entity_types": entity_types,
    }
