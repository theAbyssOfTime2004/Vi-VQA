"""The prompt conditions a checkpoint can be scored under.

Each condition differs *only* in what context sits in front of the
question. Everything else — the image, the question wording, the model,
the decoding settings — is held fixed, which is what makes the
differences between their scores readable:

    oracle-fact  −  oracle-seed-graph   what traversal and ranking lose
    oracle-seed-graph − vision-seed-graph  what vision-seeding loses
    vision-seed-graph −  no-context      what the graph is actually worth

Retrieval runs *here*, at evaluation time, not in `fvqa prepare`. Baking
retrieved facts into the split file would mean re-preparing the dataset
to change `max_hops` or swap a ranker, would make A/B-ing retrieval
settings impossible without regenerating data, and would let a result
file's recorded settings drift out of step with the prompt it actually
scored. The split keeps the raw question; the condition builds the prompt.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from fvqa.config import Config
from fvqa.data.grounding import apply_grounding
from fvqa.data.samples import IMAGE_TOKEN

__all__ = [
    "CONDITIONS",
    "ConditionContext",
    "PreparedPrompt",
    "build_prompt",
    "oracle_seed_labels",
]

logger = logging.getLogger(__name__)

#: Every condition `fvqa eval --condition` accepts.
CONDITIONS = (
    "stored",
    "no-context",
    "style",
    "oracle-fact",
    "oracle-seed-graph",
    "vision-seed-graph",
)

#: Conditions that need the knowledge graph loaded.
_GRAPH_CONDITIONS = ("oracle-fact", "oracle-seed-graph", "vision-seed-graph")

#: Conditions that actually walk the graph. `oracle-fact` needs the facts
#: loaded to look one up by id, but traverses nothing — reporting hop
#: counts and a ranker for it would describe work that never happened.
_TRAVERSAL_CONDITIONS = ("oracle-seed-graph", "vision-seed-graph")


@dataclass
class PreparedPrompt:
    """One question's messages, plus how its context was arrived at."""

    messages: list[dict[str, Any]]
    question: str
    #: The context text placed in front of the question, if any.
    context: str | None = None
    #: Retrieval provenance, absent for conditions that do not retrieve.
    retrieval: dict[str, Any] | None = None


@dataclass
class ConditionContext:
    """Everything the conditions share, built once per evaluation run.

    The graph is loaded lazily: `no-context` and `style` do not need it,
    and reading 225,434 triples takes ~10s that those runs should not pay.
    """

    config: Config
    condition: str
    #: Required by `vision-seed-graph`, which asks it what it can see.
    model: Any = None
    _graph: Any = None
    _retriever: Any = None
    _seed_provider: Any = None
    _facts: dict[str, Any] = field(default_factory=dict)
    _loaded: bool = False

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(
                f"unknown condition {self.condition!r}. Available: {list(CONDITIONS)}"
            )
        if self.condition == "vision-seed-graph" and self.model is None:
            raise ValueError(
                "the vision-seed-graph condition needs a model to ask what it sees; "
                "none was given"
            )

    @property
    def seed_provider(self) -> Any:
        """The vision seed provider, built once and cached on disk."""
        if self._seed_provider is None:
            from fvqa.retrieval import QwenVisionSeedProvider, SeedCache

            settings = self.config.retrieval
            cache = SeedCache(settings.seed_cache_dir, self.config.model.model_id)
            self._seed_provider = QwenVisionSeedProvider(self.model, cache=cache)
        return self._seed_provider

    @property
    def needs_graph(self) -> bool:
        return self.condition in _GRAPH_CONDITIONS

    @property
    def traverses(self) -> bool:
        return self.condition in _TRAVERSAL_CONDITIONS

    def _load_graph(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        from fvqa.data.fvqa import load_facts
        from fvqa.data.fvqa_graph import KnowledgeGraph
        from fvqa.retrieval import GraphRetriever

        root = self.config.data.root
        logger.info("loading the knowledge graph from %s", root)
        self._facts = load_facts(root)
        self._graph = KnowledgeGraph(self._facts)
        logger.info(
            "graph: %d triples, %d entities", len(self._graph), self._graph.num_entities
        )
        self._retriever = GraphRetriever.from_config(self._graph, self.config)

    @property
    def graph(self) -> Any:
        self._load_graph()
        return self._graph

    @property
    def retriever(self) -> Any:
        self._load_graph()
        return self._retriever

    def fact_surface(self, fact_id: str) -> str | None:
        self._load_graph()
        fact = self._facts.get(fact_id)
        return fact.get("surface") if fact else None


def _normalize(text: str) -> str:
    from fvqa.evaluation.metrics import normalize_text

    return normalize_text(text)


def oracle_seed_labels(triple: Any, answer: str) -> list[str]:
    """The supporting fact's entity labels that are *not* the answer.

    A fact links two entities; for most FVQA questions one of them is the
    answer and the other is what the question or the image is about.
    Seeding a traversal with the answer's own entity would hand the model
    the answer inside the seed and measure nothing at all, so those are
    excluded — even under the "oracle" label, the seed is allowed to be
    the correct starting point but never the correct destination.

    Returns an empty list when both endpoints look like the answer, which
    is a question this condition cannot be run on rather than an error.
    """
    normalized_answer = _normalize(answer)
    labels = []
    for label in (triple.e1_label, triple.e2_label):
        normalized = _normalize(label)
        if not normalized:
            continue
        if normalized == normalized_answer or normalized_answer in normalized:
            continue
        labels.append(label)
    return labels


def _raw_question(sample: Mapping[str, Any]) -> str:
    """The ungrounded question.

    `fvqa prepare` stores it explicitly. Splits written before it did are
    still readable: with grounding off, the human turn *is* the question.
    """
    stored = sample.get("fvqa_question")
    if stored:
        return str(stored)

    for turn in sample.get("conversations", []):
        if turn.get("from") == "human":
            value = turn.get("value", "")
            return value.replace(f"{IMAGE_TOKEN}\n", "").replace(IMAGE_TOKEN, "").strip()
    return ""


def _answer(sample: Mapping[str, Any]) -> str:
    stored = sample.get("fvqa_answer")
    if stored:
        return str(stored)
    for turn in sample.get("conversations", []):
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    return ""


def _messages(
    image_path: str,
    question: str,
    *,
    system_turns: list[str],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": text}]}
        for text in system_turns
        if text
    ]
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    )
    return messages


def _stored_prompt(
    sample: Mapping[str, Any], image_path: str, style_prompt: str
) -> PreparedPrompt:
    """Replay the prompt exactly as `fvqa prepare` wrote it.

    The condition to use on a fine-tuned checkpoint: a model trained on
    grounded prompts and queried with bare ones sees a format it never
    saw in training, and answers worse for reasons that look like a
    modelling problem rather than a plumbing one.
    """
    system_turns = [style_prompt] if style_prompt else []
    question = ""
    context = None

    for turn in sample.get("conversations", []):
        speaker = turn.get("from")
        value = turn.get("value", "")
        if speaker == "system":
            system_turns.append(value)
            context = value
        elif speaker == "human":
            question = (
                value.replace(f"{IMAGE_TOKEN}\n", "").replace(IMAGE_TOKEN, "").strip()
            )

    return PreparedPrompt(
        messages=_messages(image_path, question, system_turns=system_turns),
        question=question,
        context=context,
    )


def build_prompt(
    sample: Mapping[str, Any], context: ConditionContext
) -> PreparedPrompt:
    """Build one question's prompt under `context.condition`."""
    config = context.config
    image_path = os.path.join(config.data.image_folder, sample["image"])
    style_prompt = config.inference.system_prompt

    if context.condition == "stored":
        return _stored_prompt(sample, image_path, style_prompt)

    question = _raw_question(sample)

    if context.condition == "no-context":
        # Deliberately drops the style prompt too: this is the floor every
        # other condition is measured against, so it carries nothing.
        return PreparedPrompt(
            messages=_messages(image_path, question, system_turns=[]),
            question=question,
        )

    if context.condition == "style":
        if not style_prompt:
            logger.warning(
                "condition 'style' with an empty inference.system_prompt is identical "
                "to 'no-context' — set inference.system_prompt to measure anything"
            )
        return PreparedPrompt(
            messages=_messages(image_path, question, system_turns=[style_prompt]),
            question=question,
            context=style_prompt or None,
        )

    if context.condition == "oracle-fact":
        fact_ids = sample.get("fvqa_fact_ids") or []
        surface = context.fact_surface(fact_ids[0]) if fact_ids else None
        return _grounded(
            image_path, question, surface, style_prompt, config,
            retrieval={
                "condition": "oracle-fact",
                "status": "ok" if surface else "no_oracle_fact",
                "fact_ids": list(fact_ids),
                # True by construction when the fact resolved: this
                # condition *is* handing the supporting fact over. Leaving
                # the key out would make the run-level summary read 0%,
                # as though retrieval had failed on every question.
                "oracle_fact_retrieved": bool(surface),
            },
        )

    if context.condition == "vision-seed-graph":
        # The real pipeline: nothing from the question's annotation
        # reaches the seeding step, so the starting entity is a guess from
        # the image, with every way that can go wrong.
        return _vision_seed_graph(sample, image_path, question, style_prompt, context)

    # oracle-seed-graph: given the correct starting entity, but never the
    # correct fact — the traversal has to find it.
    return _oracle_seed_graph(sample, image_path, question, style_prompt, context)


def _grounded(
    image_path: str,
    question: str,
    context_text: str | None,
    style_prompt: str,
    config: Config,
    *,
    retrieval: dict[str, Any] | None = None,
) -> PreparedPrompt:
    """Fold context into the prompt through the shared grounding path.

    Every condition that supplies facts goes through `apply_grounding`,
    so two conditions differ only in *where the facts came from*. If they
    also differed in prompt shape, the comparison would be measuring two
    things at once.

    Grounding is forced on here regardless of `data.grounding.enabled`.
    That flag decides whether `fvqa prepare` bakes oracle facts into the
    split file; asking for a grounded *condition* is a separate decision,
    already made by naming the condition. Without this, running
    `--condition oracle-fact` under the default config (grounding off)
    would quietly hand the model no facts at all and report the score as
    if it had — the exact silent no-op these conditions exist to rule out.
    The templates and max_chars still come from the config.
    """
    grounding = replace(config.data.grounding, enabled=True)
    prompt = apply_grounding(question, context_text, grounding)
    system_turns = [style_prompt] if style_prompt else []
    if prompt.system:
        system_turns.append(prompt.system)

    return PreparedPrompt(
        messages=_messages(image_path, prompt.question, system_turns=system_turns),
        question=prompt.question,
        context=context_text,
        retrieval=retrieval,
    )


def _seeded_graph(
    sample: Mapping[str, Any],
    image_path: str,
    question: str,
    style_prompt: str,
    context: ConditionContext,
    *,
    seeds: list[str],
    source: str,
    no_seed_status: str,
) -> PreparedPrompt:
    """Retrieve from `seeds`, ground the result, record how it went.

    Shared by the oracle- and vision-seeded conditions so that the two
    differ in exactly one thing: where the seeds came from. Any other
    difference would make the gap between their scores unreadable, which
    is the only reason both conditions exist.
    """
    from fvqa.retrieval import format_facts

    config = context.config
    fact_ids = sample.get("fvqa_fact_ids") or []
    settings = config.retrieval

    if not seeds:
        # No usable seed. Recorded, not silently downgraded to a
        # no-context prompt that would score as if retrieval had run.
        return _grounded(
            image_path, question, None, style_prompt, config,
            retrieval={
                "condition": context.condition,
                "status": no_seed_status,
                "seed_texts": [],
                "resolved_entities": [],
                "facts": [],
                "oracle_fact_ids": list(fact_ids),
                "oracle_fact_retrieved": False,
                "settings": settings.as_dict(),
            },
        )

    result = context.retriever.retrieve(
        seeds,
        question,
        max_hops=settings.max_hops,
        max_seed_entities=settings.max_seed_entities,
        max_candidate_facts=settings.max_candidate_facts,
        top_k_facts=settings.top_k_facts,
        source=source,
    )

    provenance = result.as_dict()
    provenance["condition"] = context.condition
    # Whether the fact the question was actually written from survived
    # retrieval — the per-sample form of recall@k, so a result file can be
    # analysed without re-running anything.
    provenance["oracle_fact_ids"] = list(fact_ids)
    provenance["oracle_fact_retrieved"] = bool(
        fact_ids and any(f.triple.fact_id == fact_ids[0] for f in result.facts)
    )

    facts_text = format_facts(result.facts) or None
    return _grounded(
        image_path, question, facts_text, style_prompt, config, retrieval=provenance
    )


def _vision_seed_graph(
    sample: Mapping[str, Any],
    image_path: str,
    question: str,
    style_prompt: str,
    context: ConditionContext,
) -> PreparedPrompt:
    seeds = context.seed_provider.seeds(
        image_path,
        question,
        image_id=sample.get("image", ""),
        question_id=sample.get("id", ""),
    )
    return _seeded_graph(
        sample, image_path, question, style_prompt, context,
        seeds=seeds,
        source="vision",
        # Distinct from the retriever's own "no_seed_match": this means
        # the model named nothing at all, not that it named something the
        # graph does not know.
        no_seed_status="no_vision_seed",
    )


def _oracle_seed_graph(
    sample: Mapping[str, Any],
    image_path: str,
    question: str,
    style_prompt: str,
    context: ConditionContext,
) -> PreparedPrompt:
    fact_ids = sample.get("fvqa_fact_ids") or []

    seeds: list[str] = []
    if fact_ids:
        try:
            triple = context.graph.fact(fact_ids[0])
        except KeyError:
            triple = None
        if triple is not None:
            seeds = oracle_seed_labels(triple, _answer(sample))

    return _seeded_graph(
        sample, image_path, question, style_prompt, context,
        seeds=seeds,
        source="oracle",
        # Both endpoints look like the answer, or the fact id does not
        # resolve: a question this condition cannot be run on.
        no_seed_status="no_oracle_seed",
    )
