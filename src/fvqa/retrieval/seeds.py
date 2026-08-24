"""Where a traversal's starting entity comes from.

`GraphRetriever` takes seed *text*. Something has to produce that text,
and the whole point of the experiment is that on a real image it has to
come from the image. This module holds the ways of getting it, behind one
interface, so retrieval can be tested with a fixed list and run for real
with a vision model without either path knowing about the other.

The interface takes an image and a question and nothing else. That is a
guardrail, not an oversight: a seed provider that could see the
supporting fact or the answer could hand back the answer as a "guess",
and the vision-seeded condition would score high while measuring nothing.
Structurally denying it that access is worth more than remembering not to
use it.

(The oracle seed used by `--condition oracle-seed-graph` is *not* one of
these. It is derived from the question's supporting fact, not from the
image, and lives in `fvqa.evaluation.conditions` where that derivation is
visible. Dressing it up as a SeedProvider would hide the fact that it
never looks at the image at all.)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Protocol, Sequence

__all__ = [
    "SEED_PROMPT",
    "ManualSeedProvider",
    "QwenVisionSeedProvider",
    "SeedCache",
    "SeedProvider",
    "parse_seed_response",
]

logger = logging.getLogger(__name__)

#: What the model is asked. Deliberately mentions objects, scenes *and*
#: actions: FVQA's questions are tagged obj/scn/act, and a prompt asking
#: only "what object is this" cannot seed the scene and action questions
#: at all. Says nothing about the answer or about knowledge — this step
#: names what is visible, and nothing else.
SEED_PROMPT = (
    "Identify up to 5 visible objects, scenes, or actions in this image that are "
    "relevant to the following question. Name only what you can see; do not answer "
    "the question.\n\n"
    "Question: {question}\n\n"
    'Return JSON only, in exactly this form:\n{{"entities": ["...", "..."]}}'
)

MAX_SEEDS = 5

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class SeedProvider(Protocol):
    """Produces candidate starting entities for one question."""

    def seeds(self, image_path: str, question: str) -> list[str]:
        ...


class ManualSeedProvider:
    """A fixed list of seeds, for testing retrieval without a model."""

    name = "manual"

    def __init__(self, seeds: Sequence[str]):
        self._seeds = list(seeds)

    def seeds(self, image_path: str, question: str) -> list[str]:
        return list(self._seeds)


def parse_seed_response(text: str) -> list[str]:
    """Extract the entity list from whatever the model actually returned.

    Models wrap JSON in code fences, prefix it with "Here are the
    entities:", or return a bare list. Rather than trusting the format,
    this finds the outermost JSON object, and falls back to line-splitting
    when there is no parseable JSON at all — a seed list is cheap to
    recover badly and expensive to lose entirely, since losing it turns
    into a retrieval failure that looks like the graph's fault.

    Deduplicates case-insensitively, preserves order, caps at MAX_SEEDS.
    """
    entities: list[str] = []

    match = _JSON_BLOCK.search(text or "")
    if match:
        try:
            payload = json.loads(match.group(0))
            raw = payload.get("entities") if isinstance(payload, dict) else None
            if isinstance(raw, list):
                entities = [item for item in raw if isinstance(item, str)]
        except json.JSONDecodeError:
            logger.debug("seed response was not valid JSON: %r", text[:200])

    if not entities:
        # No usable JSON. Take non-empty lines, stripped of list markers.
        for line in (text or "").splitlines():
            cleaned = line.strip().strip("-*•").strip().strip('",')
            if cleaned and not cleaned.startswith(("{", "}", "[", "]")) and ":" not in cleaned:
                entities.append(cleaned)

    seen: set[str] = set()
    result: list[str] = []
    for entity in entities:
        cleaned = entity.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= MAX_SEEDS:
            break
    return result


class SeedCache:
    """Stores one question's seeds on disk, keyed by model and question.

    Without this, every experiment that touches retrieval settings — a
    different ranker, a different hop count, a different top-k — re-runs
    the vision model over the whole split to arrive at the identical
    seeds. The seeds depend on the model, the image and the question, and
    on nothing downstream of them, so they only need computing once.
    """

    def __init__(self, root: str, model_id: str):
        self.root = root
        self.model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)

    def path_for(self, image_id: str, question_id: str) -> str:
        safe_image = re.sub(r"[^A-Za-z0-9._-]+", "_", str(image_id))
        safe_question = re.sub(r"[^A-Za-z0-9._-]+", "_", str(question_id))
        return os.path.join(self.root, self.model_slug, safe_image, f"{safe_question}.json")

    def get(self, image_id: str, question_id: str) -> list[str] | None:
        path = self.path_for(image_id, question_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            # A corrupt cache entry must not end the run; recompute it.
            logger.warning("ignoring unreadable seed cache entry: %s", path)
            return None
        seeds = payload.get("seeds")
        return [s for s in seeds if isinstance(s, str)] if isinstance(seeds, list) else None

    def put(self, image_id: str, question_id: str, seeds: Sequence[str], raw: str = "") -> None:
        path = self.path_for(image_id, question_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                # The raw response is kept so a bad seed list can be traced
                # to what the model actually said, rather than to a guess
                # about how parsing went wrong.
                {"seeds": list(seeds), "raw": raw},
                handle,
                ensure_ascii=False,
                indent=2,
            )


class QwenVisionSeedProvider:
    """Asks the VLM what it can see, and uses that as the starting entity.

    This is the step that makes the pipeline real: no part of the question's
    annotation reaches it, so what it produces is a guess from the image,
    with all the ways that can go wrong.

    Args:
        model: A `VQAModel`.
        cache: Optional `SeedCache`.
        max_new_tokens: Short on purpose — the reply is a small JSON object,
            and a long budget only buys the model room to explain itself.
    """

    name = "qwen-vision"

    def __init__(
        self,
        model: Any,
        cache: SeedCache | None = None,
        max_new_tokens: int = 64,
    ):
        self.model = model
        self.cache = cache
        self.max_new_tokens = max_new_tokens

    def seeds(
        self,
        image_path: str,
        question: str,
        *,
        image_id: str | None = None,
        question_id: str | None = None,
    ) -> list[str]:
        if self.cache and image_id and question_id:
            cached = self.cache.get(image_id, question_id)
            if cached is not None:
                return cached

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": SEED_PROMPT.format(question=question)},
                ],
            }
        ]

        try:
            raw = self.model.generate(
                messages,
                max_new_tokens=self.max_new_tokens,
                # Greedy: seeds must not change between runs of an
                # otherwise identical experiment.
                temperature=0.0,
            )
        except Exception as error:  # noqa: BLE001 - one bad image must not end the run
            logger.warning("vision seeding failed for %s: %s", image_path, error)
            return []

        seeds = parse_seed_response(raw)
        if self.cache and image_id and question_id:
            self.cache.put(image_id, question_id, seeds, raw=raw)
        return seeds
