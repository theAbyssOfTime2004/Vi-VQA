"""Telling apart the kinds of thing you can be asked to load.

`fvqa eval --model-path X` accepts three different things, and they need
three different load paths:

* a HuggingFace model id (`Qwen/Qwen3-VL-8B-Instruct`) — the zero-shot
  baseline every fine-tuning result has to be read against;
* a full-weights checkpoint directory (`config.json` +
  `model*.safetensors`);
* a LoRA adapter directory (`adapter_config.json` +
  `adapter_model.safetensors`), which is what the trainer actually
  writes when `lora_enable` is on — and which is *not* loadable by
  `from_pretrained` alone. It holds only the adapter deltas; without the
  base model underneath there is nothing for them to be added to.

Two traps this module exists to avoid, both learned from reading what
`2U1/Qwen-VL-Series-Finetune` actually saves rather than from what an
adapter checkpoint is usually assumed to contain:

1. **A LoRA checkpoint from this trainer also contains `config.json`.**
   It is written by `model.config.save_pretrained(output_dir)` next to
   the adapter. So "has config.json, therefore full model" is wrong and
   would send an adapter directory down the full-weights path, where
   `from_pretrained` loads a randomly-initialised model or fails on
   missing weights. Adapter files have to be checked *first*.

2. **`non_lora_state_dict.bin` holds trained weights that are not in the
   adapter.** The trainer saves every `requires_grad` parameter that is
   not a LoRA tensor into that file — with this project's config
   (`freeze_merger: false`) that is the vision-language merger. Loading
   only `adapter_model.safetensors` therefore silently keeps the
   *untrained* merger and reports no error at all: the model runs, and
   simply scores worse than the run that produced it. The file has to be
   loaded into the base model before PEFT wraps it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "ADAPTER_CONFIG",
    "NON_LORA_STATE_DICT",
    "CheckpointInfo",
    "CheckpointType",
    "detect_checkpoint_type",
    "inspect_checkpoint",
]

logger = logging.getLogger(__name__)

CheckpointType = Literal["full", "peft", "hf_id"]

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
NON_LORA_STATE_DICT = "non_lora_state_dict.bin"

# Any one of these is enough for `AutoProcessor.from_pretrained` to have
# something to work with. A checkpoint without them has to borrow the
# base model's processor.
PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class CheckpointInfo:
    """What a `--model-path` turned out to be, and what it needs to load."""

    kind: CheckpointType
    path: str
    #: For `peft`: the base model the adapter was trained against, read
    #: from `adapter_config.json`. None for the other kinds.
    base_model_id: str | None = None
    #: For `peft`: absolute path to `non_lora_state_dict.bin` when the
    #: trainer wrote one. None means the run had nothing trainable
    #: outside the adapter — not that it is safe to skip if present.
    non_lora_path: str | None = None
    #: Whether the checkpoint carries its own processor/tokenizer config.
    has_processor: bool = False

    @property
    def is_adapter(self) -> bool:
        return self.kind == "peft"

    def processor_source(self, fallback: str | None = None) -> str:
        """Where to load the processor from.

        A checkpoint's own processor config wins: it records the chat
        template and image bounds the model was actually trained with. An
        adapter directory that lacks one falls back to the base model.
        """
        if self.has_processor:
            return self.path
        source = fallback or self.base_model_id
        if not source:
            raise ValueError(
                f"{self.path} has no processor config and no base model to fall "
                "back to. Pass the base model id explicitly."
            )
        return source


def _first_existing(directory: str, names: tuple[str, ...]) -> str | None:
    for name in names:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def detect_checkpoint_type(path: str) -> CheckpointType:
    """Classify a `--model-path` without loading anything.

    Anything that is not an existing local directory is treated as a
    HuggingFace model id and passed through untouched — that is what
    makes `--model-path Qwen/Qwen3-VL-8B-Instruct` work for the baseline.

    Raises:
        FileNotFoundError: the directory holds an `adapter_config.json`
            but no adapter weights beside it, which is a broken
            checkpoint rather than a full-weights one.
    """
    if not os.path.isdir(path):
        return "hf_id"

    if os.path.isfile(os.path.join(path, ADAPTER_CONFIG)):
        # Checked before config.json on purpose: this trainer writes both
        # into an adapter directory (see the module docstring).
        if _first_existing(path, ADAPTER_WEIGHTS) is None:
            raise FileNotFoundError(
                f"{path} contains {ADAPTER_CONFIG} but none of {list(ADAPTER_WEIGHTS)}. "
                "The adapter weights are missing — this checkpoint cannot be loaded. "
                "If training was interrupted before the first save, use an earlier "
                "checkpoint-N directory."
            )
        return "peft"

    return "full"


def inspect_checkpoint(path: str) -> CheckpointInfo:
    """Classify a checkpoint and read what loading it will need.

    Raises:
        FileNotFoundError: the adapter directory is missing its weights.
        ValueError: `adapter_config.json` is unreadable or names no base
            model, so there is nothing to load the adapter on top of.
    """
    kind = detect_checkpoint_type(path)

    if kind != "peft":
        has_processor = os.path.isdir(path) and _first_existing(path, PROCESSOR_FILES) is not None
        return CheckpointInfo(kind=kind, path=path, has_processor=has_processor)

    adapter_config_path = os.path.join(path, ADAPTER_CONFIG)
    try:
        with open(adapter_config_path, encoding="utf-8") as handle:
            adapter_config = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {adapter_config_path}: {error}") from error

    base_model_id = adapter_config.get("base_model_name_or_path")
    if not base_model_id:
        raise ValueError(
            f"{adapter_config_path} does not name a base model "
            "(`base_model_name_or_path`), so there is nothing to load the adapter "
            "onto. Pass the base model explicitly."
        )

    non_lora = os.path.join(path, NON_LORA_STATE_DICT)
    return CheckpointInfo(
        kind="peft",
        path=path,
        base_model_id=base_model_id,
        non_lora_path=non_lora if os.path.isfile(non_lora) else None,
        has_processor=_first_existing(path, PROCESSOR_FILES) is not None,
    )


def normalize_non_lora_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip the wrapper prefixes PEFT adds, so keys match the base model.

    The tensors were named while the model was wrapped in
    `get_peft_model`, which prefixes everything with `base_model.` (and
    the inner model with a further `model.`). Loading them onto the
    unwrapped base model means undoing exactly that, in the same order
    the trainer's own loader does it — a mismatch here does not raise, it
    just silently matches nothing.
    """
    stripped = {
        (key[len("base_model.") :] if key.startswith("base_model.") else key): value
        for key, value in state_dict.items()
    }
    if any(key.startswith("model.model.") for key in stripped):
        stripped = {
            (key[len("model.") :] if key.startswith("model.") else key): value
            for key, value in stripped.items()
        }
    return stripped


def load_non_lora_weights(model: Any, non_lora_path: str) -> int:
    """Load the trainer's `non_lora_state_dict.bin` into a base model.

    `strict=False` is unavoidable — the LoRA tensors live in a separate
    file and would otherwise count as missing — but it also means a key
    that matches nothing is dropped without a word. Those keys are
    trained weights, so a silent drop produces a model that runs happily
    with the untrained module and scores worse for no visible reason.
    Fail loudly instead.

    Returns:
        The number of tensors loaded.
    """
    import torch

    state_dict = torch.load(non_lora_path, map_location="cpu", weights_only=True)
    state_dict = normalize_non_lora_keys(state_dict)

    result = model.load_state_dict(state_dict, strict=False)
    unexpected = list(getattr(result, "unexpected_keys", []))
    if unexpected:
        preview = ", ".join(sorted(unexpected)[:8])
        raise RuntimeError(
            f"{len(unexpected)} of {len(state_dict)} tensors in {non_lora_path} match no "
            f"parameter of the model and would have been discarded silently: {preview}. "
            "The model would keep its untrained weights for those modules."
        )

    logger.info("loaded %d non-LoRA trained tensor(s) from %s", len(state_dict), non_lora_path)
    return len(state_dict)
