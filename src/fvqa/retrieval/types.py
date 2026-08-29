"""What retrieval produces, and what it records about producing it.

Every field here exists so a result file can answer "why did the model
see *this* fact?" months later. Retrieval settings change the score
without changing the code, so a number without its provenance —
which seeds, how many hops, which ranker — is not reproducible and not
comparable against the run next to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fvqa.data.fvqa_graph import Triple

__all__ = ["EntityCandidate", "RetrievalResult", "RetrievedFact"]


@dataclass(frozen=True)
class EntityCandidate:
    """A guessed starting point, and whether it resolved to a graph node.

    `entity_id` is None when nothing in the graph matched — which is a
    result, not an error, and has to be recorded as one. Silently
    returning an empty context instead would make a retrieval failure
    indistinguishable from a model that had the facts and still got the
    answer wrong.
    """

    text: str
    entity_id: str | None = None
    score: float = 1.0
    #: Where the guess came from: "oracle", "vision", "manual".
    source: str = "manual"

    @property
    def resolved(self) -> bool:
        return self.entity_id is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "entity_id": self.entity_id,
            "score": round(self.score, 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class RetrievedFact:
    """One fact the traversal found, with both scores kept separate.

    `retrieval_score` comes from the traversal alone — how far out the
    fact was found, before the question is looked at. `rank_score` is the
    ranker's verdict once the question is taken into account. Keeping
    both makes it possible to see whether a bad answer came from the
    graph reaching the wrong region or from the ranker mis-ordering a
    region that did contain the right fact.
    """

    triple: Triple
    hop: int
    retrieval_score: float = 0.0
    rank_score: float = 0.0

    @property
    def text(self) -> str:
        """The fact as a sentence, for putting in a prompt."""
        return self.triple.surface

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.triple.fact_id,
            "surface": self.triple.surface,
            "relation": self.triple.relation,
            "kb_source": self.triple.kb_source,
            "hop": self.hop,
            "retrieval_score": round(self.retrieval_score, 4),
            "rank_score": round(self.rank_score, 4),
        }


@dataclass
class RetrievalResult:
    """The facts chosen, plus everything needed to reproduce the choice."""

    facts: list[RetrievedFact] = field(default_factory=list)
    seeds: list[EntityCandidate] = field(default_factory=list)
    #: How many facts the traversal saw before ranking cut it to top-k.
    num_candidates: int = 0
    #: "ok", "no_seed_match", or "no_facts_found" — see GraphRetriever.
    status: str = "ok"
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def found_facts(self) -> bool:
        return bool(self.facts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "seed_texts": [seed.text for seed in self.seeds],
            "resolved_entities": [
                seed.entity_id for seed in self.seeds if seed.resolved
            ],
            "seeds": [seed.as_dict() for seed in self.seeds],
            "num_candidates": self.num_candidates,
            "facts": [fact.as_dict() for fact in self.facts],
            "settings": dict(self.settings),
        }
