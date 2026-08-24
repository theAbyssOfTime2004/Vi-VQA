"""Text seeds in, ranked facts out.

The pipeline, in one place:

    seed text ("trumpet")
      -> find_entities        resolve the guess to graph node(s)
      -> bfs_with_hops        every fact within max_hops of those nodes
      -> rank(question, ...)  order by relevance to the question
      -> top-k                what actually goes in the prompt

Deliberately takes seed *text*, not images. Vision-seeding is a separate
concern with its own failure mode, and mixing the two would mean a bad
score could never be attributed: was the graph wrong, or was the model's
guess at what it was looking at wrong? With text seeds, retrieval is
testable on its own — and running it with the question's own oracle
entity as the seed measures traversal and ranking with the vision step
taken out of the picture entirely.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Sequence

from fvqa.data.fvqa_graph import KnowledgeGraph
from fvqa.retrieval.ranker import Ranker, build_ranker
from fvqa.retrieval.types import EntityCandidate, RetrievalResult, RetrievedFact

__all__ = ["GraphRetriever"]

logger = logging.getLogger(__name__)

#: Articles a natural-language seed carries and a graph label does not.
_LEADING_ARTICLES = ("a ", "an ", "the ")


def normalize_seed(text: str) -> str:
    """Trim a seed the way a graph label is written.

    A model asked what it sees answers "a trumpet"; the node is labelled
    "trumpet". Nothing clever — lowercase, strip, drop a leading article
    — but without it a correct guess resolves to nothing.
    """
    cleaned = text.strip().lower().strip(".,;:!?\"'")
    for article in _LEADING_ARTICLES:
        if cleaned.startswith(article):
            cleaned = cleaned[len(article) :]
            break
    return cleaned.strip()


class GraphRetriever:
    """Finds the facts a question needs by walking the knowledge graph.

    Args:
        graph: The loaded knowledge graph.
        ranker: How to order candidates. Defaults to the lexical ranker.
    """

    def __init__(self, graph: KnowledgeGraph, ranker: Ranker | None = None):
        self.graph = graph
        self.ranker = ranker or build_ranker("lexical")

    @classmethod
    def from_config(cls, graph: KnowledgeGraph, config: Any) -> "GraphRetriever":
        """Build a retriever using `config.retrieval.ranking_method`."""
        return cls(graph, ranker=build_ranker(config.retrieval.ranking_method))

    def resolve_seeds(
        self,
        seed_texts: Sequence[str],
        *,
        max_seed_entities: int = 5,
        source: str = "manual",
    ) -> list[EntityCandidate]:
        """Map guessed entity names onto graph nodes.

        A guess that matches nothing is kept in the result with
        `entity_id=None` rather than dropped, so a retrieval failure
        stays visible instead of turning into an empty context that looks
        exactly like a question needing no facts.
        """
        candidates: list[EntityCandidate] = []
        seen: set[str] = set()

        for text in seed_texts:
            needle = normalize_seed(text)
            if not needle:
                continue

            matches = self.graph.find_entities(needle, limit=max_seed_entities)
            if not matches:
                candidates.append(
                    EntityCandidate(text=text, entity_id=None, score=0.0, source=source)
                )
                continue

            for rank, entity_id in enumerate(matches):
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                candidates.append(
                    EntityCandidate(
                        text=text,
                        entity_id=entity_id,
                        # Earlier matches are the exact-label ones.
                        score=1.0 / (1 + rank),
                        source=source,
                    )
                )
                if len(seen) >= max_seed_entities:
                    break
            if len(seen) >= max_seed_entities:
                break

        return candidates

    def retrieve(
        self,
        seed_texts: Sequence[str],
        question: str,
        *,
        max_hops: int = 2,
        max_seed_entities: int = 5,
        max_candidate_facts: int = 100,
        top_k_facts: int = 5,
        source: str = "manual",
    ) -> RetrievalResult:
        """Find the facts most likely to answer `question`.

        Args:
            seed_texts: Guessed entity names to start from.
            question: Used for ranking, never for traversal.
            max_hops: Traversal depth.
            max_seed_entities: Cap on resolved starting nodes.
            max_candidate_facts: Cap on what the traversal collects
                before ranking. A well-connected seed reaches a large
                slice of the graph long before `max_hops` runs out.
            top_k_facts: How many facts to keep.
            source: Recorded on each seed — "oracle", "vision", "manual".

        Returns:
            A `RetrievalResult` whose `status` is one of:

            ``ok``
                Facts were found.
            ``no_seed_match``
                No seed text resolved to any graph node — the guess was
                not something this graph knows about.
            ``no_facts_found``
                Seeds resolved, but the traversal reached no facts.
        """
        settings = {
            "max_hops": max_hops,
            "max_seed_entities": max_seed_entities,
            "max_candidate_facts": max_candidate_facts,
            "top_k_facts": top_k_facts,
            "ranking_method": getattr(self.ranker, "name", type(self.ranker).__name__),
        }

        seeds = self.resolve_seeds(
            seed_texts, max_seed_entities=max_seed_entities, source=source
        )
        entity_ids = [seed.entity_id for seed in seeds if seed.resolved]

        if not entity_ids:
            logger.info("no graph entity matched seeds %s", list(seed_texts))
            return RetrievalResult(
                facts=[], seeds=seeds, status="no_seed_match", settings=settings
            )

        candidates = self.graph.bfs_with_hops(
            entity_ids, max_hops, max_facts=max_candidate_facts
        )
        if not candidates:
            return RetrievalResult(
                facts=[], seeds=seeds, status="no_facts_found", settings=settings
            )

        scored: list[RetrievedFact] = []
        for triple, hop in candidates:
            # retrieval_score is traversal's own opinion, formed before
            # the question is read; rank_score is the ranker's, after.
            fact = RetrievedFact(triple=triple, hop=hop, retrieval_score=1.0 / hop)
            scored.append(
                replace(fact, rank_score=self.ranker.score(question, fact, seeds))
            )

        # Ties broken by hop then fact id, so an unchanged graph and an
        # unchanged question always produce the same context.
        scored.sort(key=lambda f: (-f.rank_score, f.hop, f.triple.fact_id))

        return RetrievalResult(
            facts=scored[:top_k_facts],
            seeds=seeds,
            num_candidates=len(candidates),
            status="ok",
            settings=settings,
        )
