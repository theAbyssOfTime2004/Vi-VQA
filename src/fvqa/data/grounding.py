"""Fold a piece of context text into a question, for grounded prompts.

Generic on its own — it just formats `{description}` and `{question}`
into a template — but the interesting case in this project is FVQA's
*oracle fact* grounding: `data/fvqa.py` passes the question's one correct
supporting fact (its `surface` text, e.g. "trumpets are found in jazz
clubs") as `description`, which turns "recall this from the image alone"
into "read this fact and answer". That is a different, easier condition
than the graph-retrieval path in `fvqa_graph.py`, which does not get to
see which fact is correct.

Two modes:
    prefix — the description is folded into the user turn. Works with any
             loader, which is why it is the default.
    system — the description becomes a separate `system` turn. Cleaner
             separation, but the training loader must understand a system
             role; verify before switching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fvqa.config import GroundingConfig

__all__ = ["GroundedPrompt", "apply_grounding", "truncate_description"]

# Split after ., ! or ? followed by whitespace. Deliberately simple: the
# only job is finding a readable cut point, not perfect segmentation.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def truncate_description(description: str, max_chars: int) -> str:
    """Trim a description to `max_chars`, preferring a sentence boundary.

    Cutting mid-sentence hands the model a dangling clause it may try to
    complete, so whole sentences are kept while they fit. If even the
    first sentence is too long, the text is cut hard at `max_chars` —
    a truncated fact beats an empty context.
    """
    description = description.strip()
    if len(description) <= max_chars:
        return description

    kept: list[str] = []
    length = 0
    for sentence in _SENTENCE_END.split(description):
        addition = len(sentence) + (1 if kept else 0)
        if length + addition > max_chars:
            break
        kept.append(sentence)
        length += addition

    if kept:
        return " ".join(kept).strip()
    return description[:max_chars].rstrip()


@dataclass(frozen=True)
class GroundedPrompt:
    """A question after grounding.

    Attributes:
        question: Text for the user turn (image placeholder excluded).
        system: Text for a system turn, or None when there isn't one.
    """

    question: str
    system: str | None = None


def apply_grounding(
    question: str,
    description: str | None,
    config: GroundingConfig,
) -> GroundedPrompt:
    """Build the prompt for one question.

    Falls through to the bare question when grounding is disabled or the
    record has no description, so callers need no special-casing and an
    ungrounded record never silently becomes an empty context block.
    """
    if not config.enabled or not description or not description.strip():
        return GroundedPrompt(question=question)

    context = truncate_description(description, config.max_chars)
    if not context:
        return GroundedPrompt(question=question)

    if config.mode == "system":
        # The question stays in the user turn; the system turn carries
        # context only.
        return GroundedPrompt(
            question=question,
            system=config.system_template.format(description=context),
        )
    return GroundedPrompt(
        question=config.template.format(description=context, question=question)
    )
