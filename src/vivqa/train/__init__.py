"""Training-command construction and launching."""

from vivqa.train.command import (
    REPO_URL,
    TRAIN_ENTRYPOINT,
    build_train_command,
    format_command,
    latest_checkpoint,
    resolve_model_source,
)
from vivqa.train.runner import ensure_trainer_repo, run_training

__all__ = [
    "REPO_URL",
    "TRAIN_ENTRYPOINT",
    "build_train_command",
    "ensure_trainer_repo",
    "format_command",
    "latest_checkpoint",
    "resolve_model_source",
    "run_training",
]
