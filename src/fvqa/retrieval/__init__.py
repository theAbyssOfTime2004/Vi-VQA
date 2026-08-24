"""Graph retrieval: finding the facts a question needs by walking the graph.

The counterpart to oracle-fact grounding. Oracle grounding hands the
model the one correct fact, chosen in advance; this has to find it, from
a guessed starting entity, with no knowledge of which fact is the right
one. The gap between the two scores is what the graph is actually worth.
"""

from fvqa.retrieval.context import format_facts
from fvqa.retrieval.ranker import LexicalRanker, Ranker, build_ranker
from fvqa.retrieval.retriever import (
    GraphRetriever,
    normalize_seed,
    seed_variants,
    singularize,
)
from fvqa.retrieval.seeds import (
    ManualSeedProvider,
    QwenVisionSeedProvider,
    SeedCache,
    SeedProvider,
    parse_seed_response,
)
from fvqa.retrieval.types import EntityCandidate, RetrievalResult, RetrievedFact

__all__ = [
    "EntityCandidate",
    "GraphRetriever",
    "LexicalRanker",
    "ManualSeedProvider",
    "QwenVisionSeedProvider",
    "Ranker",
    "RetrievalResult",
    "RetrievedFact",
    "SeedCache",
    "SeedProvider",
    "build_ranker",
    "format_facts",
    "normalize_seed",
    "parse_seed_response",
    "seed_variants",
    "singularize",
]
