"""The shared training-sample shape, and the pieces that operate on it
regardless of which loader produced the samples.

Every loader (`data/fvqa.py`; previously a Vi-VQA/HuggingFace loader that
lived here and was removed once the project settled on FVQA) converts its
raw records into the same JSON shape — the one `2U1/Qwen-VL-Series-Finetune`
expects:

    {"id": "...", "image": "image_0.jpg",
     "conversations": [{"from": "human", "value": "<image>\\nQ"},
                       {"from": "gpt", "value": "A"}]}

Nothing in this module reads a network or a specific dataset's schema,
which is what keeps it reusable across loaders.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from fvqa.config import SplitConfig

__all__ = ["IMAGE_TOKEN", "assign_splits", "write_split"]

logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<image>"


def assign_splits(
    image_ids: Iterable[Any],
    splits: SplitConfig,
    seed: int,
) -> dict[str, str]:
    """Map each image id to a split name.

    Splitting happens per *image*, never per QA pair — questions about the
    same photograph must never straddle the train/val boundary, or the
    model effectively sees validation images during training and
    validation loss reads better than it is.

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
