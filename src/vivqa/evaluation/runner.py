"""Run a checkpoint over a split and score it."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, Sequence

from vivqa.config import Config
from vivqa.data.prepare import IMAGE_TOKEN
from vivqa.evaluation.metrics import compute_metrics

__all__ = ["evaluate", "load_split", "messages_from_sample", "reference_of"]

logger = logging.getLogger(__name__)


def load_split(path: str) -> list[dict[str, Any]]:
    """Read a prepared split file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"split file not found: {path}. Run `vivqa prepare` first."
        )
    with open(path, "r", encoding="utf-8") as handle:
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
) -> dict[str, Any]:
    """Generate predictions for a split and score them.

    Args:
        model: A `VQAModel`.
        config: Loaded configuration.
        split: Split name. Defaults to `evaluation.split`.
        num_samples: How many samples to score; -1 for all. Defaults to
            `evaluation.num_samples`.
        output_path: Where to write the full results JSON.

    Returns:
        A dict with the metric scores and every prediction.
    """
    settings = config.evaluation
    split = split or settings.split
    num_samples = settings.num_samples if num_samples is None else num_samples

    samples = load_split(config.data.split_file(split))
    if num_samples > 0:
        samples = samples[:num_samples]

    logger.info("evaluating %d samples from the %s split", len(samples), split)

    predictions: list[str] = []
    references: list[str] = []
    records: list[dict[str, Any]] = []
    failures = 0

    for index, sample in enumerate(samples):
        reference = reference_of(sample)
        try:
            messages = messages_from_sample(
                sample,
                config.data.image_folder,
                system_prompt=config.inference.system_prompt,
            )
            prediction = model.generate(messages, temperature=settings.temperature)
        except Exception as error:  # noqa: BLE001 - one bad sample must not end the run
            logger.warning("sample %s failed: %s", sample.get("id", index), error)
            failures += 1
            continue

        predictions.append(prediction)
        references.append(reference)
        records.append(
            {
                "id": sample.get("id"),
                "image": sample.get("image"),
                "question": messages[-1]["content"][-1]["text"],
                "reference": reference,
                "prediction": prediction,
            }
        )

        if (index + 1) % 25 == 0:
            logger.info("scored %d/%d", index + 1, len(samples))

    scores = compute_metrics(predictions, references, settings.metrics)

    result = {
        "split": split,
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
        f"Knowledge grounding: {'on' if result.get('grounding_enabled') else 'off'}",
        "=" * 60,
    ]
    for name, value in result["metrics"].items():
        # CIDEr lives on 0-10; the rest are percentages.
        suffix = "" if name == "cider" else "%"
        lines.append(f"  {name:<12} {value:8.2f}{suffix}")
    lines.append("=" * 60)
    return "\n".join(lines)
