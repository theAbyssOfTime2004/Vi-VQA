"""Fetch the trainer repository and launch a run."""

from __future__ import annotations

import logging
import os
import subprocess

from vivqa.config import Config
from vivqa.train.command import REPO_URL, TRAIN_ENTRYPOINT, build_train_command, format_command

__all__ = ["ensure_trainer_repo", "run_training"]

logger = logging.getLogger(__name__)


def ensure_trainer_repo(dest: str, url: str = REPO_URL) -> str:
    """Clone `2U1/Qwen-VL-Series-Finetune` into `dest` if it isn't there.

    Returns:
        Absolute path to the repository.

    Raises:
        RuntimeError: the clone failed, or the directory exists but does
            not contain the training entry point.
    """
    dest = os.path.abspath(dest)

    if not os.path.exists(dest):
        logger.info("cloning %s into %s", url, dest)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, dest],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{result.stderr.strip()}")

    entrypoint = os.path.join(dest, TRAIN_ENTRYPOINT)
    if not os.path.isfile(entrypoint):
        raise RuntimeError(
            f"{dest} does not contain {TRAIN_ENTRYPOINT}. "
            "Delete the directory and let it be cloned again."
        )
    return dest


def run_training(
    config: Config,
    *,
    trainer_dir: str = "./Qwen-VL-Series-Finetune",
    num_gpus: int = 1,
    resume: bool = False,
    dry_run: bool = False,
) -> int:
    """Launch fine-tuning. Returns the trainer's exit code.

    Paths are made absolute before the working directory changes to the
    trainer repository, so a relative `data_dir` in the config still
    resolves against the directory the command was run from.
    """
    data = config.data
    train_path = os.path.abspath(data.split_file("train"))
    val_path = os.path.abspath(data.split_file("val"))
    image_folder = os.path.abspath(data.image_folder)
    output_dir = os.path.abspath(config.training.output_dir)

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"training data not found at {train_path}. Run `vivqa prepare` first."
        )
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"image folder not found: {image_folder}")

    if not os.path.exists(val_path):
        logger.warning("no validation split at %s; evaluation during training is off", val_path)
        val_path = None

    repo = ensure_trainer_repo(trainer_dir)
    os.makedirs(output_dir, exist_ok=True)

    command = build_train_command(
        config,
        train_path=train_path,
        val_path=val_path,
        image_folder=image_folder,
        output_dir=output_dir,
        num_gpus=num_gpus,
        resume=resume,
    )

    logger.info("training command:\n%s", format_command(command))
    if dry_run:
        return 0

    result = subprocess.run(command, cwd=repo)
    if result.returncode != 0:
        logger.error("training exited with code %d", result.returncode)
    return result.returncode
