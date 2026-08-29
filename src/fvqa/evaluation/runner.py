"""Run a checkpoint over a split and score it."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

from fvqa.config import Config
from fvqa.data.samples import IMAGE_TOKEN
from fvqa.evaluation.conditions import CONDITIONS, ConditionContext, build_prompt
from fvqa.evaluation.metrics import compute_metrics

__all__ = [
    "CONDITIONS",
    "evaluate",
    "load_split",
    "messages_from_sample",
    "reference_of",
]

logger = logging.getLogger(__name__)


def load_split(path: str) -> list[dict[str, Any]]:
    """Read a prepared split file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"split file not found: {path}. Run `fvqa prepare` first."
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def messages_from_sample(
    sample: Mapping[str, Any],
    image_folder: str,
    system_prompt: str = "",
) -> list[dict[str, Any]]:
    """Rebuild the chat messages stored in a prepared sample.

    The prompt is replayed exactly as written into the split file — any
    grounding it carries included — so evaluation measures the model on
    the prompt format it was trained on.

    `system_prompt` is prepended as a style-only turn. It is what makes a
    prompt-engineered baseline measurable: the same stored questions, the
    same model, one instruction about answer format and nothing else. It
    carries no information about the image, so a score change is
    attributable to style alone.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    for turn in sample["conversations"]:
        speaker = turn.get("from")
        value = turn.get("value", "")

        if speaker == "system":
            messages.append({"role": "system", "content": [{"type": "text", "text": value}]})
        elif speaker == "human":
            question = value.replace(f"{IMAGE_TOKEN}\n", "").replace(IMAGE_TOKEN, "").strip()
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": os.path.join(image_folder, sample["image"])},
                        {"type": "text", "text": question},
                    ],
                }
            )
        # The assistant turn is the reference answer, not part of the prompt.

    return messages


def reference_of(sample: Mapping[str, Any]) -> str:
    """The ground-truth answer stored in a sample."""
    for turn in sample["conversations"]:
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    return ""


def evaluate(
    model: Any,
    config: Config,
    *,
    split: str | None = None,
    num_samples: int | None = None,
    output_path: str | None = None,
    condition: str = "stored",
) -> dict[str, Any]:
    """Generate predictions for a split and score them.

    Args:
        model: A `VQAModel`.
        config: Loaded configuration.
        split: Split name. Defaults to `evaluation.split`.
        num_samples: How many samples to score; -1 for all. Defaults to
            `evaluation.num_samples`.
        output_path: Where to write the full results JSON.
        condition: Which prompt condition to score under — see
            `fvqa.evaluation.conditions.CONDITIONS`. The default replays
            the prompt stored in the split file, which is what a
            fine-tuned checkpoint was trained on.

    Returns:
        A dict with the metric scores and every prediction.
    """
    settings = config.evaluation
    split = split or settings.split
    num_samples = settings.num_samples if num_samples is None else num_samples

    # The model is handed to the context because `vision-seed-graph` has
    # to ask it what it can see before the question is even posed.
    condition_context = ConditionContext(
        config=config, condition=condition, model=model
    )

    samples = load_split(config.data.split_file(split))
    if num_samples > 0:
        samples = samples[:num_samples]

    logger.info(
        "evaluating %d samples from the %s split (condition: %s)",
        len(samples), split, condition,
    )

    predictions: list[str] = []
    references: list[str] = []
    records: list[dict[str, Any]] = []
    failures = 0

    for index, sample in enumerate(samples):
        reference = reference_of(sample)
        try:
            prompt = build_prompt(sample, condition_context)
            prediction = model.generate(
                prompt.messages, temperature=settings.temperature
            )
        except Exception as error:  # noqa: BLE001 - one bad sample must not end the run
            logger.warning("sample %s failed: %s", sample.get("id", index), error)
            failures += 1
            continue

        predictions.append(prediction)
        references.append(reference)
        record = {
            "id": sample.get("id"),
            "image": sample.get("image"),
            "question": prompt.question,
            "reference": reference,
            "prediction": prediction,
        }
        if prompt.retrieval is not None:
            # Per-sample provenance: which seeds were tried, what resolved,
            # which facts reached the prompt. Without it a low score cannot
            # be split between "retrieval missed the fact" and "the model
            # had the fact and still got it wrong".
            record["retrieval"] = prompt.retrieval
        records.append(record)

        if (index + 1) % 25 == 0:
            logger.info("scored %d/%d", index + 1, len(samples))

    scores = compute_metrics(predictions, references, settings.metrics)

    result = {
        "split": split,
        "condition": condition,
        "num_samples": len(predictions),
        "num_failed": failures,
        "grounding_enabled": config.data.grounding.enabled,
        # Recorded so two result files can be told apart later: a score is
        # meaningless without knowing which prompt produced it.
        "system_prompt": config.inference.system_prompt,
        "max_new_tokens": config.inference.max_new_tokens,
        "metrics": scores,
        "predictions": records,
    }

    if condition_context.traverses:
        result["retrieval"] = config.retrieval.as_dict()
    if condition_context.needs_graph:
        result["retrieval_summary"] = summarize_retrieval(records)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        logger.info("wrote results to %s", output_path)

    return result


def format_scores(result: Mapping[str, Any]) -> str:
    """Render evaluation results as a readable block."""
    lines = [
        "=" * 60,
        f"Evaluation — split={result['split']} n={result['num_samples']}"
        + (f" (failed: {result['num_failed']})" if result.get("num_failed") else ""),
        f"Condition: {result.get('condition', 'stored')}",
    ]

    summary = result.get("retrieval_summary")
    if summary:
        settings = result.get("retrieval")
        if settings:
            lines.append(
                f"Retrieval: {settings.get('max_hops')} hop(s), "
                f"top-{settings.get('top_k_facts')}, {settings.get('ranking_method')}"
            )
        # The upper bound on what grounding could have contributed: the
        # model cannot use a fact retrieval never handed it.
        lines.append(
            f"  supporting fact retrieved: {summary['oracle_fact_retrieved']}"
            f"/{summary['num_with_provenance']}"
            f" ({summary['recall']:.1%})"
        )
        if summary["failed_retrievals"]:
            lines.append(f"  retrieval failures: {summary['failed_retrievals']}")

    lines.append("=" * 60)
    for name, value in result["metrics"].items():
        # CIDEr lives on 0-10; the rest are percentages.
        suffix = "" if name == "cider" else "%"
        lines.append(f"  {name:<12} {value:8.2f}{suffix}")
    lines.append("=" * 60)
    return "\n".join(lines)


def summarize_retrieval(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample retrieval provenance into run-level numbers.

    `recall` is the share of scored questions whose own supporting fact
    survived retrieval into the prompt. It bounds what grounding could
    possibly have contributed — a fact the model was never handed cannot
    have helped it — so reading a metric without it invites crediting the
    graph for answers it played no part in.
    """
    with_provenance = [r["retrieval"] for r in records if r.get("retrieval")]
    retrieved = sum(1 for r in with_provenance if r.get("oracle_fact_retrieved"))
    failures = sum(1 for r in with_provenance if r.get("status") not in ("ok", None))

    return {
        "num_with_provenance": len(with_provenance),
        "oracle_fact_retrieved": retrieved,
        "recall": retrieved / len(with_provenance) if with_provenance else 0.0,
        "failed_retrievals": failures,
    }
