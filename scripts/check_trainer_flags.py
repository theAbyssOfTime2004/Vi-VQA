#!/usr/bin/env python3
"""Verify build_train_command()'s flags against the real trainer.

`fvqa/train/command.py` hand-writes `--flag value` pairs for
`2U1/Qwen-VL-Series-Finetune`'s `HfArgumentParser`. Nothing catches a
flag name drifting from that repo's actual dataclasses until a training
run fails several minutes in with "unrecognized arguments". This script
is that check: it enumerates the flags every code path in
`build_train_command` can emit, and confirms each one is a real field on
the trainer's `ModelArguments` / `DataArguments` / `TrainingArguments`.

The trainer's own field names are read with `ast`, not by importing it —
that repo pulls in torch/trl/accelerate just to define those dataclasses.
Its `TrainingArguments` inherits most of its fields from
`transformers.TrainingArguments`, and those are read from the *installed*
transformers when it is importable.

That last part is the whole check, and an earlier version got it wrong.
It used a hardcoded list of "standard HuggingFace fields that have been
stable for years", which reported all flags fine while the run died on
`--warmup_ratio`: transformers 5 removed it (keeping `warmup_steps`).
An allowlist cannot notice a field disappearing upstream, which is the
one thing this check exists to notice. When transformers is not
installed the fallback list is still used, and the output says the check
was partial rather than implying it was complete.

Run this by hand — nothing runs it automatically. The two moments it can
tell you anything are when `trainer.revision` is bumped and when
`build_train_command()` is edited, so it belongs next to those changes
rather than on a timer:

    python scripts/check_trainer_flags.py <path-to-cloned-trainer-repo>

Exits non-zero (and prints exactly which flags are wrong) on a mismatch.
This caught a real bug once already: command.py sent `--eval_data_path`
but the trainer's field is `eval_path`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Used only when transformers is not installed, to say something rather
# than nothing. Necessarily out of date the moment upstream changes —
# `warmup_ratio` sat in this list while transformers 5 no longer had it.
_FALLBACK_HF_TRAINING_ARGS = {
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
    "warmup_steps",
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


def _hf_training_arg_names() -> tuple[set[str], bool]:
    """Fields of the installed `transformers.TrainingArguments`.

    Returns:
        (field names, whether they came from the real transformers)
    """
    try:
        import dataclasses as _dc

        from transformers import TrainingArguments
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return set(_FALLBACK_HF_TRAINING_ARGS), False
    return {field.name for field in _dc.fields(TrainingArguments)}, True


def _trainer_field_names(trainer_src: Path) -> tuple[set[str], bool]:
    params_py = trainer_src / "params.py"
    if not params_py.is_file():
        raise FileNotFoundError(f"{params_py} not found")
    source = params_py.read_text(encoding="utf-8")
    own_fields = _dataclass_field_names(
        source, {"ModelArguments", "DataArguments", "TrainingArguments"}
    )
    inherited, exact = _hf_training_arg_names()
    return own_fields | inherited, exact


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

    known, exact = _trainer_field_names(trainer_src)
    emitted = _emitted_flags()
    unknown = sorted(emitted - known)

    if not exact:
        print(
            "NOTE: transformers is not installed here, so inherited Trainer fields "
            "come from a fallback list and this check is partial. Run it in the "
            "environment that will do the training for a real answer.\n"
        )

    if unknown:
        print("FAIL: build_train_command() emits flag(s) the trainer does not accept:")
        for flag in unknown:
            print(f"  --{flag}")
        print(
            "\nCheck the field names in "
            f"{trainer_src}/params.py (ModelArguments/DataArguments/TrainingArguments) "
            "and in the installed transformers.TrainingArguments, "
            "against src/fvqa/train/command.py."
        )
        return 1

    source = "installed transformers" if exact else "a fallback list"
    print(
        f"OK: all {len(emitted)} flags build_train_command() can emit are recognized "
        f"(inherited fields from {source})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
