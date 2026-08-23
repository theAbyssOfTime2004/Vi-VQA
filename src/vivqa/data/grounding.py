"""Knowledge grounding from the dataset's own `description` field.

Each record in Viet-ViTextVQA-gemini-VQA carries a ~558-character
description of the image alongside the QA turns. Gemini wrote the answers
*from that description*, so the answers routinely assert facts that are
not legible in the pixels — the year a temple was built, which district a
market is in. Training on (image, question) alone asks the model to
recall knowledge it was never shown; feeding the description back turns
the same sample into a reading-comprehension task it can actually learn.

The description is knowledge the dataset already paid for, and until now
no code path in this repository touched it.

Two modes:
    prefix — the description is folded into the user turn. Works with any
             loader, which is why it is the default.
    system — the description becomes a separate `system` turn. Cleaner
             separation, but the training loader must understand a system
             role; verify before switching (see docs/GROUNDING.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vivqa.config import GroundingConfig

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
