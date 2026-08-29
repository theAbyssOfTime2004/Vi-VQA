"""Fetch the trainer repository and launch a run."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

from fvqa.config import Config
from fvqa.train.command import REPO_URL, TRAIN_ENTRYPOINT, build_train_command, format_command

__all__ = ["ensure_trainer_repo", "run_training"]

logger = logging.getLogger(__name__)


def _run_git(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, **kwargs)


def ensure_trainer_repo(dest: str, url: str = REPO_URL, revision: str | None = None) -> str:
    """Clone `2U1/Qwen-VL-Series-Finetune` into `dest`, pinned to `revision`.

    Without a pin, upstream moving `HEAD` (a flag rename, a restructure)
    can silently break `build_train_command`'s output between two runs
    that used the identical local config. With `revision` set, a fresh
    `dest` is fetched at exactly that commit (a shallow fetch of the SHA,
    falling back to a full clone when the remote refuses to serve an
    unadvertised commit directly); an already-cloned `dest` left on a
    different commit gets a warning, not a silent re-checkout, since a
    human working directory may have local changes worth not clobbering.

    Returns:
        Absolute path to the repository.

    Raises:
        RuntimeError: the clone/checkout failed, or the directory exists
            but does not contain the training entry point.
    """
    dest = os.path.abspath(dest)

    if not os.path.exists(dest):
        if revision:
            logger.info("fetching %s @ %s into %s", url, revision[:12], dest)
            os.makedirs(dest, exist_ok=True)
            _run_git(["init", "-q", dest])
            _run_git(["-C", dest, "remote", "add", "origin", url])
            fetched = _run_git(["-C", dest, "fetch", "--depth", "1", "origin", revision])
            if fetched.returncode == 0:
                checkout = _run_git(["-C", dest, "checkout", "-q", "FETCH_HEAD"])
                if checkout.returncode != 0:
                    raise RuntimeError(f"git checkout failed:\n{checkout.stderr.strip()}")
            else:
                # Some git servers refuse to serve an unadvertised commit
                # SHA over a shallow fetch. Fall back to a full clone.
                logger.info("shallow fetch of pinned commit failed, doing a full clone")
                shutil.rmtree(dest)
                cloned = _run_git(["clone", url, dest])
                if cloned.returncode != 0:
                    raise RuntimeError(f"git clone failed:\n{cloned.stderr.strip()}")
                checkout = _run_git(["-C", dest, "checkout", "-q", revision])
                if checkout.returncode != 0:
                    raise RuntimeError(f"git checkout failed:\n{checkout.stderr.strip()}")
        else:
            logger.info("cloning %s into %s (no revision pinned)", url, dest)
            cloned = _run_git(["clone", "--depth", "1", url, dest])
            if cloned.returncode != 0:
                raise RuntimeError(f"git clone failed:\n{cloned.stderr.strip()}")
    elif revision:
        current = _run_git(["-C", dest, "rev-parse", "HEAD"])
        head = current.stdout.strip() if current.returncode == 0 else "?"
        if not head.startswith(revision) and not revision.startswith(head):
            logger.warning(
                "trainer repo at %s is on %s, not the pinned revision %s — "
                "delete the directory to have it re-fetched at the pin",
                dest, head[:12], revision[:12],
            )

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
            f"training data not found at {train_path}. Run `fvqa prepare` first."
        )
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"image folder not found: {image_folder}")

    if not os.path.exists(val_path):
        logger.warning("no validation split at %s; evaluation during training is off", val_path)
        val_path = None

    repo = ensure_trainer_repo(
        trainer_dir, url=config.trainer.repo_url, revision=config.trainer.revision
    )
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
