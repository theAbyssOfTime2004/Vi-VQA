"""Turning retrieved facts into the text a prompt can carry.

Kept separate from the retriever so that *what was found* and *how it is
worded to the model* stay independent: changing the wording must not
require re-running retrieval, and comparing two prompt formats over the
identical retrieved facts has to be possible.

The output feeds `apply_grounding` as its `description`, which is what
makes graph retrieval and oracle-fact grounding go through the same
prompt-assembly path. Two conditions that differ only in where the facts
came from is the comparison this project is built to make; two conditions
that also differ in prompt shape would not be one.
"""

from __future__ import annotations

from collections.abc import Sequence

from fvqa.retrieval.types import RetrievedFact

__all__ = ["format_facts"]


def format_facts(facts: Sequence[RetrievedFact], *, numbered: bool = True) -> str:
    """Render facts as the context string for a grounded prompt.

    Returns an empty string for no facts — the caller then builds an
    ungrounded prompt, which is the honest representation of "retrieval
    found nothing" rather than a prompt with an empty Facts: header.
    """
    if not facts:
        return ""

    lines = []
    for index, fact in enumerate(facts, start=1):
        text = fact.text.strip()
        lines.append(f"{index}. {text}" if numbered else text)
    return "\n".join(lines)
