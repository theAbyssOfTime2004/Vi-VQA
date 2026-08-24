"""Graph retrieval: finding the facts a question needs by walking the graph.

The counterpart to oracle-fact grounding. Oracle grounding hands the
model the one correct fact, chosen in advance; this has to find it, from
a guessed starting entity, with no knowledge of which fact is the right
one. The gap between the two scores is what the graph is actually worth.
"""

from fvqa.retrieval.context import format_facts
from fvqa.retrieval.ranker import LexicalRanker, Ranker, build_ranker
from fvqa.retrieval.retriever import GraphRetriever, normalize_seed
from fvqa.retrieval.types import EntityCandidate, RetrievalResult, RetrievedFact

__all__ = [
    "EntityCandidate",
    "GraphRetriever",
    "LexicalRanker",
    "Ranker",
    "RetrievalResult",
    "RetrievedFact",
    "build_ranker",
    "format_facts",
    "normalize_seed",
]
