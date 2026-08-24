"""Loader for FVQA (Fact-based Visual Question Answering).

FVQA ships a real knowledge graph: 225,434 triples from DBpedia, ConceptNet
and WebChild, and 5,826 questions each anchored to exactly one supporting
fact. Unlike Viet-ViTextVQA, the "knowledge" here is not free text — it is
`(e1, relation, e2)` triples you can build a graph from and traverse.
`data/fvqa_graph.py` does the traversal; this module only gets the raw
release into the shape the rest of `vivqa` expects.

Schema notes below come from downloading the release and inspecting it
directly (2026-08-24), not from the README, which understates two counts
(193,449 facts / 5,286 questions vs. the 225,434 / 5,826 actually shipped)
and omits that `fact` is a list and that `e1`/`e2` are heterogeneous IDs
(ConceptNet URIs, DBpedia URLs, or bare lowercase words for WebChild,
depending on `KB`) rather than a single uniform identifier scheme.

Download:
    https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=1
    (~451MB; verified reachable 2026-08-24)

Layout after extraction:
    Name_Lists/{train,test}_list_{0..4}.txt   5 official train/test folds,
                                               one image filename per line
    new_dataset_release/images/               2,190 images (COCO .jpg +
                                               ImageNet .JPEG, mixed case)
    new_dataset_release/all_fact_triples_release.json   225,434 facts,
                                               keyed by a KB-specific ID
    new_dataset_release/all_qs_dict_release.json        5,826 questions,
                                               keyed by question_id
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, Mapping

from vivqa.config import Config, FvqaConfig, SplitConfig
from vivqa.data.grounding import apply_grounding
from vivqa.data.prepare import IMAGE_TOKEN, assign_splits, write_split

__all__ = [
    "FACTS_FILENAME",
    "IMAGES_DIRNAME",
    "QUESTIONS_FILENAME",
    "build_sample",
    "image_path",
    "load_facts",
    "load_questions",
    "load_split_images",
    "prepare_fvqa",
]

logger = logging.getLogger(__name__)

FACTS_FILENAME = "new_dataset_release/all_fact_triples_release.json"
QUESTIONS_FILENAME = "new_dataset_release/all_qs_dict_release.json"
IMAGES_DIRNAME = "new_dataset_release/images"
NAME_LISTS_DIRNAME = "Name_Lists"


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Download and extract the FVQA release first: "
            "https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=1"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_facts(root: str) -> dict[str, dict[str, Any]]:
    """Load the 225,434 knowledge-graph triples, keyed by fact ID."""
    return _read_json(os.path.join(root, FACTS_FILENAME))


def load_questions(root: str) -> dict[str, dict[str, Any]]:
    """Load the 5,826 questions, keyed by question ID."""
    return _read_json(os.path.join(root, QUESTIONS_FILENAME))


def load_split_images(root: str, fold: int, split: str) -> set[str]:
    """Image filenames in one of FVQA's 5 official train/test folds.

    Args:
        fold: 0-4.
        split: "train" or "test".
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = os.path.join(root, NAME_LISTS_DIRNAME, f"{split}_list_{fold}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def image_path(root: str, img_file: str) -> str:
    """Absolute-ish path to an image file. Extensions are mixed case
    (COCO ships `.jpg`, ImageNet ships `.JPEG`) — pass `img_file` through
    unchanged rather than normalizing the extension.
    """
    return os.path.join(root, IMAGES_DIRNAME, img_file)


def build_sample(
    question_id: str,
    question_record: Mapping[str, Any],
    facts: Mapping[str, dict[str, Any]],
    config: FvqaConfig,
    grounding_config: Any,
) -> dict[str, Any]:
    """Build one training sample in the format the rest of vivqa expects.

    `fact` is a list with exactly one element for every question in the
    release (verified: 5,826/5,826), so `facts[0]` is safe without a
    length check — but this reads facts[0] rather than assuming that
    invariant holds in data nobody has re-verified since, in case a
    future release relaxes it.

    Grounding here means the *oracle* fact: `apply_grounding` receives
    the exact supporting fact's `surface` text, i.e. "what if the model
    were simply told the right fact". That is a different, easier
    condition than the graph-retrieval condition in `fvqa_graph.py`,
    which does not get to see the answer's fact ID.
    """
    fact_ids = question_record.get("fact") or []
    fact_surface = None
    if fact_ids:
        fact = facts.get(fact_ids[0])
        if fact is not None:
            fact_surface = fact.get("surface")
    # Fall back to the question's own copy of the surface text if the fact
    # ID could not be resolved — every ID was resolvable at inspection
    # time, but a loader should not hard-fail a whole prepare run over one
    # dangling reference.
    if fact_surface is None:
        fact_surface = question_record.get("fact_surface")

    question_text = question_record["question"].strip()
    answer_text = str(question_record["answer"]).strip()

    prompt = apply_grounding(question_text, fact_surface, grounding_config)

    turns: list[dict[str, str]] = []
    if prompt.system:
        turns.append({"from": "system", "value": prompt.system})
    turns.append({"from": "human", "value": f"{IMAGE_TOKEN}\n{prompt.question}"})
    turns.append({"from": "gpt", "value": answer_text})

    return {
        "id": f"fvqa_{question_id}",
        "image": question_record["img_file"],
        "image_id": question_record["img_file"],
        "conversations": turns,
        # Carried through for the graph-retrieval eval path: the oracle
        # fact ID and the visual-concept category (obj/scn/act) that a
        # traversal experiment needs and free-text grounding does not.
        "fvqa_fact_ids": fact_ids,
        "fvqa_visual_concept": question_record.get("visual_concept"),
    }


def prepare_fvqa(
    config: Config,
    *,
    limit: int | None = None,
) -> dict[str, int]:
    """Build train/val/test JSON files from a local FVQA release.

    Unlike `data.prepare.prepare`, this never downloads or re-encodes
    images: FVQA ships ready-to-use JPEGs, so samples just reference the
    existing filenames.

    FVQA's own `test_list_{fold}` is kept intact as the `test` split — it
    is what published FVQA results are measured against, so silently
    reshuffling it would make scores incomparable to the paper. `val` is
    carved out of FVQA's `train_list_{fold}` using the same
    `assign_splits` the HuggingFace path uses, renormalizing
    `data.splits.train`/`val` to ignore `.test` (FVQA already fixed that).

    Args:
        config: Loaded configuration. `config.data.fvqa` selects the root
            directory and fold.
        limit: Process at most this many questions. For smoke tests.

    Returns:
        Sample counts per split, plus `questions` (total processed).
    """
    fvqa = config.data.fvqa
    facts = load_facts(fvqa.root)
    questions = load_questions(fvqa.root)
    train_images = load_split_images(fvqa.root, fvqa.fold, "train")
    test_images = load_split_images(fvqa.root, fvqa.fold, "test")

    logger.info(
        "FVQA fold %d: %d facts, %d questions, %d train images, %d test images",
        fvqa.fold, len(facts), len(questions), len(train_images), len(test_images),
    )

    items: Iterable[tuple[str, dict[str, Any]]] = questions.items()
    if limit is not None:
        items = list(items)[:limit]

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    # Only train-fold samples need to wait for assign_splits; test-fold
    # samples have a known destination immediately.
    train_samples_by_image: dict[str, list[dict[str, Any]]] = {}

    for question_id, record in items:
        image = record.get("img_file")
        if image not in test_images and image not in train_images:
            logger.warning("question %s references an unknown-split image %s", question_id, image)
            continue

        sample = build_sample(question_id, record, facts, fvqa, config.data.grounding)
        if image in test_images:
            by_split["test"].append(sample)
        else:
            train_samples_by_image.setdefault(image, []).append(sample)

    # Renormalize train/val, ignoring the config's `test` weight — FVQA's
    # own test fold already fixed that split.
    train_weight = config.data.splits.train
    val_weight = config.data.splits.val
    total = train_weight + val_weight
    sub_splits = SplitConfig(
        train=train_weight / total if total else 1.0,
        val=val_weight / total if total else 0.0,
        test=0.0,
    )
    assignment = assign_splits(train_samples_by_image, sub_splits, config.seed)
    for image, split_name in assignment.items():
        by_split[split_name].extend(train_samples_by_image[image])

    counts: dict[str, int] = {"questions": len(items)}
    for split_name, samples in by_split.items():
        samples.sort(key=lambda s: s["id"])
        path = config.data.split_file(split_name)
        write_split(samples, path)
        counts[split_name] = len(samples)
        logger.info("wrote %d samples to %s", len(samples), path)

    return counts
