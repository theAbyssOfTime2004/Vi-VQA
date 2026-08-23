"""Build the fine-tuning command from the configuration.

Training itself is delegated to `2U1/Qwen-VL-Series-Finetune`. This module
is the single place that translates `config/config.yaml` into that
trainer's command line, so the local script and the Modal pipeline cannot
drift apart the way they had.

One correctness note carried over from the old scripts: the entry point is
`src/train/train_sft.py` launched through `deepspeed`, not the `train.py`
at the repository root. `train.py` does not exist there, so
`scripts/train_qwen3vl.sh` and the Colab notebook both failed on their
first line of real work.
"""

from __future__ import annotations

import os
import re
from typing import Sequence

from vivqa.config import Config

__all__ = [
    "REPO_URL",
    "TRAIN_ENTRYPOINT",
    "build_train_command",
    "latest_checkpoint",
    "resolve_model_source",
]

REPO_URL = "https://github.com/2U1/Qwen-VL-Series-Finetune.git"
TRAIN_ENTRYPOINT = "src/train/train_sft.py"

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _flag(value: bool) -> str:
    """The trainer parses booleans as strings, so pass 'True'/'False'."""
    return "True" if value else "False"


def latest_checkpoint(output_dir: str) -> str | None:
    """Newest `checkpoint-N` directory in `output_dir`, or None.

    Ordered by step number, not lexicographically: sorting strings puts
    `checkpoint-1000` before `checkpoint-500` and would resume training
    from the wrong place.
    """
    if not os.path.isdir(output_dir):
        return None

    checkpoints: list[tuple[int, str]] = []
    for entry in os.listdir(output_dir):
        match = _CHECKPOINT_RE.match(entry)
        if match and os.path.isdir(os.path.join(output_dir, entry)):
            checkpoints.append((int(match.group(1)), entry))

    if not checkpoints:
        return None
    return os.path.join(output_dir, max(checkpoints)[1])


def resolve_model_source(
    *,
    model_path: str | None = None,
    checkpoint: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Decide which model to load for inference or evaluation.

    Priority: an explicit `model_path` wins, then a named checkpoint under
    `output_dir`, then the newest checkpoint there.

    `model_path` is passed through untouched so a HuggingFace model id
    works alongside a local directory. That is what makes it possible to
    score the un-finetuned base model — the baseline every fine-tuning
    result has to be read against.

    Raises:
        ValueError: nothing was given to resolve from.
        FileNotFoundError: the named or newest checkpoint does not exist.
    """
    if model_path:
        return model_path

    if not output_dir:
        raise ValueError(
            "pass model_path, or output_dir to resolve a checkpoint from"
        )

    if checkpoint:
        path = os.path.join(output_dir, checkpoint)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    latest = latest_checkpoint(output_dir)
    if not latest:
        raise FileNotFoundError(
            f"no checkpoint under {output_dir}. Pass model_path to evaluate a "
            "base model, e.g. Qwen/Qwen3-VL-8B-Instruct"
        )
    return latest


def build_train_command(
    config: Config,
    *,
    train_path: str,
    image_folder: str,
    output_dir: str,
    val_path: str | None = None,
    num_gpus: int = 1,
    resume: bool = False,
    deepspeed_config: str | None = None,
) -> list[str]:
    """Translate the configuration into the trainer's argv.

    Args:
        config: Loaded configuration.
        train_path: Path to `train.json`.
        image_folder: Directory holding the extracted images.
        output_dir: Where checkpoints are written.
        val_path: Path to `val.json`. Evaluation is disabled when absent.
        num_gpus: GPUs for the deepspeed launcher.
        resume: Resume from the newest checkpoint in `output_dir`.
        deepspeed_config: Overrides `training.deepspeed`.

    Returns:
        argv ready for `subprocess.run`.
    """
    model = config.model
    training = config.training

    ds_config = deepspeed_config if deepspeed_config is not None else training.deepspeed

    if ds_config:
        command = ["deepspeed", f"--num_gpus={num_gpus}", TRAIN_ENTRYPOINT]
    else:
        # Without deepspeed there is nothing for the launcher to shard.
        command = ["python", TRAIN_ENTRYPOINT]

    command += [
        "--model_id", model.model_id,
        "--data_path", train_path,
        "--image_folder", image_folder,
        "--output_dir", output_dir,
        "--remove_unused_columns", "False",
    ]

    if ds_config:
        command += ["--deepspeed", ds_config]

    # Evaluation needs a validation file; asking for eval without one makes
    # the trainer fail several minutes in.
    if val_path:
        command += [
            "--eval_data_path", val_path,
            "--eval_strategy", training.eval_strategy,
            "--eval_steps", str(training.eval_steps),
            "--load_best_model_at_end", _flag(training.load_best_model_at_end),
            "--metric_for_best_model", training.metric_for_best_model,
            "--greater_is_better", _flag(training.greater_is_better),
        ]
    else:
        command += ["--eval_strategy", "no"]

    if model.lora.enabled:
        lora = model.lora
        command += [
            "--lora_enable", "True",
            "--use_dora", "False",
            "--lora_rank", str(lora.rank),
            "--lora_alpha", str(lora.alpha),
            "--lora_dropout", str(lora.dropout),
            "--num_lora_modules", str(lora.num_lora_modules),
            "--lora_namespan_exclude", str(lora.namespan_exclude),
            "--vision_lora", _flag(lora.vision_lora),
        ]
    else:
        command += ["--lora_enable", "False"]

    if model.qlora.enabled:
        # 4-bit base weights with adapters on top.
        command += ["--bits", "4"]

    command += [
        "--freeze_vision_tower", _flag(training.freeze_vision_tower),
        "--freeze_llm", _flag(training.freeze_llm),
        "--freeze_merger", _flag(training.freeze_merger),

        "--bf16", _flag(training.bf16),
        "--fp16", _flag(training.fp16),
        "--tf32", _flag(training.tf32),

        "--num_train_epochs", str(training.num_train_epochs),
        "--per_device_train_batch_size", str(training.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(training.per_device_eval_batch_size),
        "--gradient_accumulation_steps", str(training.gradient_accumulation_steps),

        "--image_min_pixels", str(model.image_min_pixels),
        "--image_max_pixels", str(model.image_max_pixels),

        "--learning_rate", str(training.learning_rate),
        "--vision_lr", str(training.vision_lr),
        "--merger_lr", str(training.merger_lr),
        "--lr_scheduler_type", training.lr_scheduler_type,
        "--weight_decay", str(training.weight_decay),
        "--warmup_ratio", str(training.warmup_ratio),
        "--max_grad_norm", str(training.max_grad_norm),
        "--optim", training.optim,

        "--gradient_checkpointing", _flag(training.gradient_checkpointing),
        "--dataloader_num_workers", str(training.dataloader_num_workers),
        "--lazy_preprocess", _flag(training.lazy_preprocess),

        "--save_strategy", training.save_strategy,
        "--save_steps", str(training.save_steps),
        "--save_total_limit", str(training.save_total_limit),

        "--logging_steps", str(training.logging_steps),
        "--report_to", training.report_to,

        # The trainer takes the negated form of the config flag.
        "--disable_flash_attn2", _flag(not model.use_flash_attn),
    ]

    if resume:
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint:
            command += ["--resume_from_checkpoint", checkpoint]

    return command


def format_command(command: Sequence[str]) -> str:
    """Render argv as a readable multi-line shell command, for logging."""
    parts: list[str] = [command[0]]
    index = 1
    while index < len(command):
        token = command[index]
        if token.startswith("--") and index + 1 < len(command) and not command[index + 1].startswith("--"):
            parts.append(f"{token} {command[index + 1]}")
            index += 2
        else:
            parts.append(token)
            index += 1
    return " \\\n    ".join(parts)
