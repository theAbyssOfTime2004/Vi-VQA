"""Ordering candidate facts by how well they answer the question.

BFS returns facts by graph distance, which is not the same as relevance.
A seed like "dog" has hundreds of 1-hop facts — `dog IsA animal`,
`dog AtLocation house`, `dog CapableOf barking`, `dog RelatedTo leash` —
and the question decides which of them matters. Handing all of them to
the model is not retrieval; it is giving up and hoping the context window
sorts it out.

The default ranker is lexical on purpose. It is deterministic, needs no
extra model, runs in microseconds and is straightforward to unit-test,
which makes it a baseline the fancier options have to beat rather than a
placeholder for them. An embedding ranker would handle paraphrase better
but turns the experiment into vector retrieval — the very thing this
project chose a real graph in order not to be measuring. An LLM reranker
is slower, harder to reproduce, and risks the reranking step deducing the
answer on its own, which would quietly move the thing being measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from fvqa.evaluation.metrics import tokenize
from fvqa.retrieval.types import EntityCandidate, RetrievedFact

__all__ = ["LexicalRanker", "Ranker", "build_ranker"]

# Words that appear in almost every question and would otherwise dominate
# the overlap score, since a fact mentioning "what" tells you nothing.
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being do does did what which who whom
    whose when where why how this that these those there here it its of to
    in on at for from by with about as and or if can could would should
    """.split()
)


class Ranker(Protocol):
    def score(
        self,
        question: str,
        fact: RetrievedFact,
        seeds: Sequence[EntityCandidate] = (),
    ) -> float:
        ...


class LexicalRanker:
    """Token overlap between the question and the fact, minus a hop penalty.

    The score has three parts:

    * **overlap** — how much of the question's content vocabulary the
      fact's text covers. Normalized by question length so a long
      question does not simply outscore a short one.
    * **seed bonus** — a fact that actually mentions the entity the seed
      resolved to is more likely to be about the thing in the image than
      one two steps removed from it that happens to share a word.
    * **hop penalty** — linear in distance. A 2-hop fact has to be
      noticeably more on-topic than a 1-hop one to outrank it.

    Args:
        hop_penalty: Subtracted per hop beyond the first.
        seed_bonus: Added when the fact touches a resolved seed entity.
    """

    name = "lexical"

    def __init__(self, hop_penalty: float = 0.15, seed_bonus: float = 0.25):
        self.hop_penalty = hop_penalty
        self.seed_bonus = seed_bonus

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {token for token in tokenize(text) if token not in _STOPWORDS}

    def score(
        self,
        question: str,
        fact: RetrievedFact,
        seeds: Sequence[EntityCandidate] = (),
    ) -> float:
        question_tokens = self._content_tokens(question)

        triple = fact.triple
        # The relation is part of the evidence: a question asking "what is
        # it made of" should favour a `MadeOf` edge over an `IsA` one.
        fact_text = " ".join(
            (triple.e1_label, triple.relation, triple.e2_label, triple.surface)
        )
        fact_tokens = self._content_tokens(fact_text)

        if question_tokens and fact_tokens:
            overlap = len(question_tokens & fact_tokens) / len(question_tokens)
        else:
            overlap = 0.0

        score = overlap
        seed_ids = {seed.entity_id for seed in seeds if seed.resolved}
        if seed_ids & {triple.e1, triple.e2}:
            score += self.seed_bonus

        score -= self.hop_penalty * max(0, fact.hop - 1)
        return score


def build_ranker(method: str) -> Ranker:
    """Construct the ranker named in `retrieval.ranking_method`.

    Raises:
        ValueError: no ranker goes by that name.
    """
    if method == "lexical":
        return LexicalRanker()
    raise ValueError(
        f"unknown ranking_method {method!r}. Available: 'lexical'"
    )
