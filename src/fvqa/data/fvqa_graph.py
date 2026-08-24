"""A real knowledge graph over FVQA's 225,434 triples, and traversal on it.

This is the part that does not exist anywhere else in `fvqa`: the oracle
grounding path (`fvqa.build_sample`) hands the model a piece of text that
was chosen for it in advance — the question's own correct fact. Here,
nothing is chosen in advance — `KnowledgeGraph` builds a real node/edge
structure from the triples, and `bfs`/`shortest_path` walk it themselves.

Two distinct experiments this enables, and they answer different
questions:

    oracle fact (fvqa.py)      "if the model is simply told the one
                                correct fact, how much does it help?"
    graph retrieval (here)     "if the model has to find that fact itself
                                by walking the graph from a guessed
                                starting entity, how much is lost?"

The gap between the two is the real measure of whether the graph is
pulling its weight.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

__all__ = ["Triple", "KnowledgeGraph"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Triple:
    """One `(e1, relation, e2)` edge, carrying the fields needed to both
    traverse and to render the fact as text for grounding."""

    fact_id: str
    e1: str
    e2: str
    e1_label: str
    e2_label: str
    relation: str
    kb_source: str
    surface: str

    def other(self, entity_id: str) -> str:
        """The endpoint that is *not* `entity_id`.

        Raises:
            ValueError: `entity_id` is neither endpoint of this triple.
        """
        if entity_id == self.e1:
            return self.e2
        if entity_id == self.e2:
            return self.e1
        raise ValueError(f"{entity_id!r} is not an endpoint of {self.fact_id!r}")


class KnowledgeGraph:
    """An undirected adjacency structure over FVQA's triples.

    Undirected on purpose: FVQA's `e1`/`e2` order reflects how the source
    KB (DBpedia/ConceptNet/WebChild) happened to state the fact, not a
    traversal direction a question implies. "X is taller than Y" is just
    as useful for answering a question that starts from Y as from X — the
    graph should be walkable either way. `Triple.other()` still returns
    the direction-preserving endpoint, so callers who do care which side
    was `e1` can recover it.

    Node identity is opaque here on purpose: ConceptNet nodes look like
    `/c/en/dog/n`, DBpedia nodes like `http://dbpedia.org/resource/Dog`,
    WebChild nodes are bare lowercase words. Comparing them by identity
    (not surface similarity) never accidentally merges an unrelated pair
    that happens to share a string.
    """

    def __init__(self, facts: Mapping[str, Mapping[str, Any]]):
        self._triples: dict[str, Triple] = {}
        self._adjacency: dict[str, list[str]] = {}
        # Case-folded label -> entity ids, for find_entities(); an entity
        # can carry more than one surface label across different facts.
        self._labels: dict[str, set[str]] = {}

        for fact_id, fact in facts.items():
            triple = Triple(
                fact_id=fact_id,
                e1=fact["e1"],
                e2=fact["e2"],
                e1_label=fact.get("e1_label", ""),
                e2_label=fact.get("e2_label", ""),
                relation=fact.get("r", ""),
                kb_source=fact.get("KB", ""),
                surface=fact.get("surface", ""),
            )
            self._triples[fact_id] = triple
            self._adjacency.setdefault(triple.e1, []).append(fact_id)
            self._adjacency.setdefault(triple.e2, []).append(fact_id)
            if triple.e1_label:
                self._labels.setdefault(triple.e1_label.lower(), set()).add(triple.e1)
            if triple.e2_label:
                self._labels.setdefault(triple.e2_label.lower(), set()).add(triple.e2)

    @classmethod
    def from_facts_file(cls, path: str) -> "KnowledgeGraph":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def __len__(self) -> int:
        return len(self._triples)

    @property
    def num_entities(self) -> int:
        return len(self._adjacency)

    def fact(self, fact_id: str) -> Triple:
        return self._triples[fact_id]

    def has_entity(self, entity_id: str) -> bool:
        return entity_id in self._adjacency

    def neighbors(self, entity_id: str) -> list[Triple]:
        """Every triple touching `entity_id`. Empty list if it is not a node."""
        return [self._triples[fid] for fid in self._adjacency.get(entity_id, [])]

    def find_entities(self, text: str, limit: int = 5) -> list[str]:
        """Entity IDs whose label matches `text` (case-insensitive).

        Exact label matches first, then labels containing `text` as a
        substring. This is intentionally a simple lookup, not an entity
        linker: on a real image the "starting entity" has to come from
        somewhere (an object detector, a VLM's own guess at what's
        pictured, a human) — `find_entities` only resolves that guess's
        *text* to a graph node once you already have one.
        """
        needle = text.strip().lower()
        if not needle:
            return []

        exact = self._labels.get(needle, set())
        if exact:
            return sorted(exact)[:limit]

        matches: set[str] = set()
        for label, entities in self._labels.items():
            if needle in label:
                matches.update(entities)
                if len(matches) >= limit:
                    break
        return sorted(matches)[:limit]

    def bfs(
        self,
        start_entities: Sequence[str],
        max_hops: int,
        *,
        max_facts: int | None = None,
    ) -> list[Triple]:
        """Every fact reachable within `max_hops` of any start entity.

        Breadth-first, so a fact touching a 1-hop entity is always
        returned before one only reachable at 2 hops — callers that want
        "the most relevant few facts" can just take a prefix of the
        result instead of re-sorting it.

        Args:
            start_entities: Seed node IDs. Unknown IDs are skipped rather
                than raising, since a caller resolving a guessed entity
                name through `find_entities` may pass none that exist.
            max_hops: How many edges out from the seeds to explore.
            max_facts: Stop early once this many facts are collected.
                Without it a well-connected seed can pull in a large
                fraction of the graph well before `max_hops` is reached.

        Returns:
            Triples in BFS order, each appearing once even if reachable
            through multiple paths.
        """
        if max_hops < 1:
            raise ValueError(f"max_hops must be at least 1, got {max_hops}")

        visited_entities: set[str] = set()
        visited_facts: set[str] = set()
        result: list[Triple] = []

        queue: deque[tuple[str, int]] = deque()
        for entity in start_entities:
            if entity in self._adjacency and entity not in visited_entities:
                visited_entities.add(entity)
                queue.append((entity, 0))

        while queue:
            entity, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for triple in self.neighbors(entity):
                if triple.fact_id not in visited_facts:
                    visited_facts.add(triple.fact_id)
                    result.append(triple)
                    if max_facts is not None and len(result) >= max_facts:
                        return result
                neighbor = triple.other(entity)
                if neighbor not in visited_entities:
                    visited_entities.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return result

    def shortest_path(self, start: str, goal: str) -> list[Triple] | None:
        """The shortest chain of facts connecting `start` to `goal`.

        Standard BFS parent-pointer reconstruction. Useful mainly to
        validate the graph mechanics: given a 2-hop question's own two
        entities, does the traversal actually find a path between them,
        and does it match the fact the question was built from?

        Returns:
            An ordered list of triples from `start` to `goal`, or None if
            they are in different connected components. Empty list if
            `start == goal`.
        """
        if start == goal:
            return []
        if start not in self._adjacency or goal not in self._adjacency:
            return None

        parent: dict[str, tuple[str, Triple]] = {}
        visited = {start}
        queue: deque[str] = deque([start])

        while queue:
            entity = queue.popleft()
            for triple in self.neighbors(entity):
                neighbor = triple.other(entity)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parent[neighbor] = (entity, triple)
                if neighbor == goal:
                    path: list[Triple] = []
                    node = goal
                    while node != start:
                        prev, edge = parent[node]
                        path.append(edge)
                        node = prev
                    path.reverse()
                    return path
                queue.append(neighbor)

        return None

    def iter_triples(self) -> Iterator[Triple]:
        return iter(self._triples.values())
