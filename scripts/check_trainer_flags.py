#!/usr/bin/env python3
"""Verify build_train_command()'s flags against the real trainer.

`fvqa/train/command.py` hand-writes `--flag value` pairs for
`2U1/Qwen-VL-Series-Finetune`'s `HfArgumentParser`. Nothing catches a
flag name drifting from that repo's actual dataclasses until a training
run fails several minutes in with "unrecognized arguments". This script
is that check: it enumerates the flags every code path in
`build_train_command` can emit, and confirms each one is a real field on
the trainer's `ModelArguments` / `DataArguments` / `TrainingArguments`.

Field names for `ModelArguments`/`DataArguments`/`TrainingArguments` are
read with `ast`, not by importing the trainer — that repo pulls in
torch/transformers/trl/accelerate just to define those dataclasses, and
this check has no reason to pay for that install. `TrainingArguments`
inherits most of its fields from `transformers.TrainingArguments`
(HFTrainingArguments) rather than declaring them itself, so those
inherited names are a small hardcoded allowlist below — standard
HuggingFace Trainer fields that have been stable for years. A rename on
that side is `transformers`' problem to flag in their own changelog, not
something this project's flag contract needs to re-derive.

Usage:
    python scripts/check_trainer_flags.py <path-to-cloned-trainer-repo>

Exits non-zero (and prints exactly which flags are wrong) on a mismatch.
This caught a real bug once already: command.py sent `--eval_data_path`
but the trainer's field is `eval_path`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Standard transformers.TrainingArguments fields that
# 2U1/Qwen-VL-Series-Finetune's TrainingArguments inherits rather than
# redeclares, and that build_train_command() relies on.
_KNOWN_HF_TRAINING_ARGS = {
    "output_dir",
    "remove_unused_columns",
    "deepspeed",
    "eval_strategy",
    "eval_steps",
    "load_best_model_at_end",
    "metric_for_best_model",
    "greater_is_better",
    "bf16",
    "fp16",
    "tf32",
    "num_train_epochs",
    "max_steps",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "lr_scheduler_type",
    "weight_decay",
    "warmup_ratio",
    "max_grad_norm",
    "optim",
    "gradient_checkpointing",
    "dataloader_num_workers",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "logging_steps",
    "report_to",
    "resume_from_checkpoint",
}


def _dataclass_field_names(source: str, class_names: set[str]) -> set[str]:
    """Names assigned directly in the body of the given classes (ast, no import)."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in class_names:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    names.add(stmt.target.id)
    return names


def _trainer_field_names(trainer_src: Path) -> set[str]:
    params_py = trainer_src / "params.py"
    if not params_py.is_file():
        raise FileNotFoundError(f"{params_py} not found")
    source = params_py.read_text(encoding="utf-8")
    own_fields = _dataclass_field_names(
        source, {"ModelArguments", "DataArguments", "TrainingArguments"}
    )
    return own_fields | _KNOWN_HF_TRAINING_ARGS


def _emitted_flags() -> set[str]:
    """Every `--flag` build_train_command can emit, across its branches."""
    from fvqa.config import Config
    from fvqa.train.command import TRAIN_ENTRYPOINT, build_train_command

    flags: set[str] = set()

    for tuning_method in ("full", "lora", "qlora"):
        for val_path in ("/data/val.json", None):
            for deepspeed in ("scripts/zero2.json", None):
                config = Config()
                config.model.tuning_method = tuning_method
                config.training.deepspeed = deepspeed
                # max_steps is emitted only when set, so exercise it too.
                config.training.max_steps = 2 if deepspeed else None
                command = build_train_command(
                    config,
                    train_path="/data/train.json",
                    image_folder="/data/images",
                    output_dir="/checkpoints",
                    val_path=val_path,
                )
                # Everything before and including the entry point belongs to
                # the launcher (`deepspeed --num_gpus=N ...` or `python`),
                # not to the trainer's own HfArgumentParser.
                entry_index = command.index(TRAIN_ENTRYPOINT)
                trainer_argv = command[entry_index + 1 :]
                flags |= {token[2:] for token in trainer_argv if token.startswith("--")}

    return flags


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    trainer_root = Path(sys.argv[1]).resolve()
    trainer_src = trainer_root / "src"

    known = _trainer_field_names(trainer_src)
    emitted = _emitted_flags()
    unknown = sorted(emitted - known)

    if unknown:
        print("FAIL: build_train_command() emits flag(s) the trainer does not accept:")
        for flag in unknown:
            print(f"  --{flag}")
        print(
            "\nCheck the field names in "
            f"{trainer_src}/params.py (ModelArguments/DataArguments/TrainingArguments) "
            "against src/fvqa/train/command.py."
        )
        return 1

    print(f"OK: all {len(emitted)} flags build_train_command() can emit are recognized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
