"""Turn the HuggingFace dataset into training files.

This module replaces four near-identical copies of the same logic that
had drifted apart (`src/dataset_vlm.py`, the Modal pipeline, and two
notebooks). Everything that does not touch the network or the filesystem
is a pure function, so the interesting parts are testable without
downloading 9,594 images.

Output format is the one `2U1/Qwen-VL-Series-Finetune` expects:

    {"id": "...", "image": "image_0.jpg",
     "conversations": [{"from": "human", "value": "<image>\\nQ"},
                       {"from": "gpt", "value": "A"}]}
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Iterable, Mapping, Sequence

from vivqa.config import Config, DataConfig, SplitConfig
from vivqa.data.grounding import apply_grounding

__all__ = [
    "IMAGE_TOKEN",
    "assign_splits",
    "build_samples",
    "extract_qa_pairs",
    "image_filename",
    "load_records",
    "prepare",
    "write_split",
]

logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<image>"

_USER_ROLES = frozenset({"user", "human"})
_ASSISTANT_ROLES = frozenset({"assistant", "gpt"})


def image_filename(record_id: Any) -> str:
    """Filename an image is stored under. Stable across runs."""
    return f"image_{record_id}.jpg"


def extract_qa_pairs(conversations: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """Pull (question, answer) pairs out of a multi-turn conversation.

    Handles both key spellings the dataset uses (`role`/`content` and
    `from`/`value`). A question with no following answer is dropped, as is
    an answer with no preceding question — a malformed turn should cost
    one sample, not derail the run.
    """
    pairs: list[tuple[str, str]] = []
    pending_question: str | None = None

    for turn in conversations:
        role = turn.get("role", turn.get("from"))
        content = turn.get("content", turn.get("value"))
        if role is None or content is None:
            continue

        if role in _USER_ROLES:
            pending_question = content
        elif role in _ASSISTANT_ROLES and pending_question is not None:
            question = pending_question.strip()
            answer = content.strip()
            # An empty side makes an untrainable sample; skip the pair but
            # still consume the question so turns stay aligned.
            if question and answer:
                pairs.append((question, answer))
            pending_question = None

    return pairs


def build_samples(
    record_id: Any,
    conversations: Sequence[Mapping[str, Any]],
    description: str | None,
    config: DataConfig,
) -> list[dict[str, Any]]:
    """Build the training samples for one image."""
    filename = image_filename(record_id)
    samples: list[dict[str, Any]] = []

    for index, (question, answer) in enumerate(extract_qa_pairs(conversations)):
        prompt = apply_grounding(question, description, config.grounding)

        turns: list[dict[str, str]] = []
        if prompt.system:
            turns.append({"from": "system", "value": prompt.system})
        turns.append({"from": "human", "value": f"{IMAGE_TOKEN}\n{prompt.question}"})
        turns.append({"from": "gpt", "value": answer})

        samples.append(
            {
                # Indexed within the image, not by a global counter, so an
                # id identifies the same QA pair on every run.
                "id": f"{record_id}_{index}",
                "image": filename,
                "image_id": str(record_id),
                "conversations": turns,
            }
        )

    return samples


def assign_splits(
    image_ids: Iterable[Any],
    splits: SplitConfig,
    seed: int,
) -> dict[str, str]:
    """Map each image id to a split name.

    Splitting happens per *image*, never per QA pair. The previous
    pipeline shuffled QA pairs before splitting, which put different
    questions about the same photograph on both sides of the train/val
    boundary — the model met the validation images during training and
    validation loss read better than it was.

    Sorting before shuffling makes the assignment reproducible regardless
    of the order the dataset arrives in.
    """
    unique_ids = sorted({str(image_id) for image_id in image_ids})
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    total = len(unique_ids)
    num_val = int(total * splits.val)
    num_test = int(total * splits.test)
    # Train absorbs the rounding remainder rather than dropping samples.
    num_train = total - num_val - num_test

    assignment: dict[str, str] = {}
    for image_id in unique_ids[:num_train]:
        assignment[image_id] = "train"
    for image_id in unique_ids[num_train : num_train + num_val]:
        assignment[image_id] = "val"
    for image_id in unique_ids[num_train + num_val :]:
        assignment[image_id] = "test"
    return assignment


def write_split(samples: Sequence[Mapping[str, Any]], path: str) -> None:
    """Write one split to JSON, creating parent directories as needed."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(samples), handle, ensure_ascii=False, indent=2)


def _save_image(image: Any, path: str, quality: int) -> bool:
    """Write a PIL image as RGB JPEG. Returns False if it could not be saved."""
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(path, format="JPEG", quality=quality)
        return True
    except Exception as error:  # noqa: BLE001 - one bad image must not kill the run
        logger.warning("failed to save image to %s: %s", path, error)
        return False


def load_records(data: DataConfig, limit: int | None = None) -> Iterable[Mapping[str, Any]]:
    """Fetch the dataset records.

    With `data.streaming` the split is walked lazily instead of downloaded
    in full first. Combined with `limit` that is the difference between a
    few hundred MB and several GB — which is what makes a baseline run fit
    on a Colab or Kaggle disk.
    """
    from datasets import load_dataset  # imported lazily: heavy dependency

    if data.streaming:
        if limit is None:
            # Streaming without a limit still walks every record; it only
            # saves the up-front download. Worth saying so, because the
            # run will not look any faster.
            logger.warning(
                "streaming without --limit still iterates the whole split; "
                "it only avoids the up-front download"
            )
        logger.info("streaming %s (limit=%s)", data.dataset_name, limit)
        return load_dataset(data.dataset_name, split="train", streaming=True)

    logger.info("loading %s", data.dataset_name)
    return load_dataset(data.dataset_name, split="train")


def prepare(
    config: Config,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    records: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    """Download, convert and split the dataset.

    Args:
        config: Loaded configuration.
        limit: Process at most this many records. Useful for smoke tests.
        overwrite: Re-encode images that are already on disk.
        records: Pre-loaded records, bypassing the HuggingFace download.
            Used by tests and by callers that already hold the dataset.

    Returns:
        Number of samples written per split, plus `images` and `records`.
    """
    data = config.data
    os.makedirs(data.image_folder, exist_ok=True)

    if records is None:
        records = load_records(data, limit=limit)

    samples_by_image: dict[str, list[dict[str, Any]]] = {}
    images_written = 0
    records_seen = 0

    for index, record in enumerate(records):
        if limit is not None and records_seen >= limit:
            break
        records_seen += 1

        conversations = record.get("conversations") or []
        if not conversations:
            logger.warning("record %s has no conversations, skipping", index)
            continue

        record_id = record.get("id", index)
        path = os.path.join(data.image_folder, image_filename(record_id))

        image = record.get("image")
        if image is not None and (overwrite or not os.path.exists(path)):
            if not _save_image(image, path, data.image_quality):
                continue
            images_written += 1

        samples = build_samples(
            record_id=record_id,
            conversations=conversations,
            description=record.get("description"),
            config=data,
        )
        if samples:
            samples_by_image.setdefault(str(record_id), []).extend(samples)

    assignment = assign_splits(samples_by_image, data.splits, config.seed)

    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for image_id, samples in samples_by_image.items():
        grouped[assignment[image_id]].extend(samples)

    counts: dict[str, int] = {}
    for split, samples in grouped.items():
        # Skip a split configured to 0.0 rather than writing an empty file
        # that later looks like a failed preparation step.
        if not samples and getattr(data.splits, split) == 0.0:
            continue
        # Sort for a byte-stable file: dict iteration order follows
        # insertion, which a different dataset ordering would change.
        samples.sort(key=lambda sample: sample["id"])
        path = data.split_file(split)
        write_split(samples, path)
        counts[split] = len(samples)
        logger.info("wrote %d samples to %s", len(samples), path)

    counts["images"] = images_written
    counts["records"] = records_seen
    return counts
