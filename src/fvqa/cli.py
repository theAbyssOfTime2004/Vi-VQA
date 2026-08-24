"""Command-line interface: `fvqa <command>`.

Every subcommand loads `config/config.yaml`, so a hyperparameter is
changed in one place and takes effect everywhere. `--set` applies ad-hoc
overrides without editing the file:

    fvqa prepare --set data.grounding.enabled=true
    fvqa train --set training.num_train_epochs=3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Sequence

from fvqa.config import ConfigError, load_config
from fvqa.utils import set_seed, setup_logging

logger = logging.getLogger("fvqa")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to config.yaml (default: config/config.yaml)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set training.learning_rate=1e-5",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fvqa", description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("config", help="Print the resolved configuration")
    _add_common(show)

    prepare = subparsers.add_parser("prepare", help="Build train/val/test files from a local FVQA release")
    _add_common(prepare)
    prepare.add_argument("--limit", type=int, help="Process at most this many questions")

    train = subparsers.add_parser("train", help="Fine-tune Qwen3-VL with LoRA")
    _add_common(train)
    train.add_argument("--num-gpus", type=int, default=1)
    train.add_argument("--resume", action="store_true", help="Resume from the newest checkpoint")
    train.add_argument("--trainer-dir", default="./Qwen-VL-Series-Finetune")
    train.add_argument("--dry-run", action="store_true", help="Print the command and exit")

    evaluate = subparsers.add_parser("eval", help="Score a checkpoint on a split")
    _add_common(evaluate)
    evaluate.add_argument(
        "--model-path",
        required=True,
        help="A model id, a full-weights checkpoint, or a LoRA adapter directory",
    )
    evaluate.add_argument(
        "--base-model",
        help="Base model for a LoRA adapter, overriding the one named in "
        "adapter_config.json (needed when it points at a path that has moved)",
    )
    evaluate.add_argument("--split", help="Split to score (default: evaluation.split)")
    evaluate.add_argument("--num-samples", type=int, help="-1 for the whole split")
    evaluate.add_argument("--output", help="Where to write results JSON")

    chat = subparsers.add_parser("chat", help="Ask questions about an image interactively")
    _add_common(chat)
    chat.add_argument("--model-path", required=True)
    chat.add_argument("--base-model", help="Base model for a LoRA adapter (see `eval`)")
    chat.add_argument("--image", help="Skip the image prompt and use this file for every question")

    return parser


def _command_config(args, config) -> int:
    print(json.dumps(config.as_dict(), ensure_ascii=False, indent=2))
    return 0


def _command_prepare(args, config) -> int:
    from fvqa.data.fvqa import prepare_fvqa

    counts = prepare_fvqa(config, limit=args.limit)
    print("\nPrepared FVQA dataset")
    print(f"  questions read   {counts.pop('questions', 0)}")
    for split, count in counts.items():
        print(f"  {split:<16} {count} samples -> {config.data.split_file(split)}")
    print(f"  fold             {config.data.fold}")
    print(f"  grounding        {'on (oracle fact)' if config.data.grounding.enabled else 'off'}")
    return 0


def _command_train(args, config) -> int:
    from fvqa.train.runner import run_training

    return run_training(
        config,
        trainer_dir=args.trainer_dir,
        num_gpus=args.num_gpus,
        resume=args.resume,
        dry_run=args.dry_run,
    )


def _command_eval(args, config) -> int:
    from fvqa.evaluation.runner import evaluate, format_scores
    from fvqa.model import VQAModel

    model = VQAModel.from_pretrained(
        args.model_path, config, base_model_id=args.base_model
    )
    result = evaluate(
        model,
        config,
        split=args.split,
        num_samples=args.num_samples,
        output_path=args.output,
    )
    print(format_scores(result))
    return 0


def _command_chat(args, config) -> int:
    from fvqa.model import VQAModel

    model = VQAModel.from_pretrained(
        args.model_path, config, base_model_id=args.base_model
    )
    grounding_on = config.data.grounding.enabled

    print("\nFVQA interactive. Ctrl-D or 'quit' to exit.")
    if grounding_on:
        print("Grounding is on: you will be asked for optional context after each question.")

    image_path = args.image
    while True:
        try:
            if not args.image:
                image_path = input("\nImage path: ").strip()
                if image_path.lower() in {"quit", "exit"}:
                    return 0
            if not image_path or not os.path.exists(image_path):
                print(f"  image not found: {image_path}")
                continue

            question = input("Question: ").strip()
            if question.lower() in {"quit", "exit"}:
                return 0
            if not question:
                continue

            description = None
            if grounding_on:
                description = input("Context (blank for none): ").strip() or None

            print(f"\n  {model.answer(image_path, question, description)}")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0


_COMMANDS = {
    "config": _command_config,
    "prepare": _command_prepare,
    "train": _command_train,
    "eval": _command_eval,
    "chat": _command_chat,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    try:
        config = load_config(args.config, overrides=args.overrides)
    except ConfigError as error:
        logger.error("configuration error: %s", error)
        return 2

    set_seed(config.seed)
    try:
        return _COMMANDS[args.command](args, config)
    except ImportError as error:
        logger.error(
            "%s\nThis command needs the optional dependencies. "
            "Install them with: pip install -e '.[train]'",
            error,
        )
        return 1
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
