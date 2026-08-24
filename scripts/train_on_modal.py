"""FVQA training pipeline on Modal.

All dataset, configuration and command-building logic lives in the
`fvqa` package and is shared with the local run — this file contains
only what is genuinely Modal-specific: the container image, the volumes
and the function boundaries.

Usage:
    pip install modal
    modal token new
    modal secret create huggingface-secret HF_TOKEN=hf_...   # for the model only; FVQA itself needs no auth

    modal run scripts/train_on_modal.py --step prepare
    modal run scripts/train_on_modal.py --step train
    modal run scripts/train_on_modal.py --step evaluate
    modal run scripts/train_on_modal.py --step baseline
    modal run scripts/train_on_modal.py::check_status

FVQA is much smaller than a typical VQA fine-tuning corpus — 2,190 images,
5,826 questions — so a run here is proportionally shorter than a dataset
with tens of thousands of QA pairs would take on the same GPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("fvqa-training")

FVQA_ZIP_URL = "https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=1"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget", "unzip")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        # Qwen3VLForConditionalGeneration does not exist before 4.57.
        "transformers>=4.57.0",
        "accelerate>=0.34.0",
        "peft>=0.13.0",
        "bitsandbytes>=0.44.0",
        "deepspeed>=0.15.0",
        "qwen-vl-utils>=0.0.8",
        "pyyaml>=6.0",
        "tensorboard",
    )
    .env({"CUDA_HOME": "/usr/local/cuda", "FVQA_CONFIG": "/root/config/config.yaml"})
    .run_commands("pip install flash-attn --no-build-isolation || echo 'flash-attn unavailable'")
    # Ship the package and config instead of duplicating their logic here.
    .add_local_dir(REPO_ROOT / "src" / "fvqa", remote_path="/root/fvqa")
    .add_local_dir(REPO_ROOT / "config", remote_path="/root/config")
)

data_volume = modal.Volume.from_name("fvqa-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("fvqa-checkpoints", create_if_missing=True)

DATA_DIR = "/data"
CHECKPOINT_DIR = "/checkpoints"
FVQA_ROOT = f"{DATA_DIR}/fvqa"
VOLUMES = {DATA_DIR: data_volume, CHECKPOINT_DIR: checkpoint_volume}

# Point the shared config at the mounted volumes. Everything else —
# learning rates, LoRA rank, grounding — still comes from config.yaml.
VOLUME_OVERRIDES = [
    f"data.data_dir={DATA_DIR}",
    f"data.root={FVQA_ROOT}",
    f"data.image_folder={FVQA_ROOT}/new_dataset_release/images",
    f"training.output_dir={CHECKPOINT_DIR}/qwen3vl-fvqa",
]


def _load(extra_overrides: list[str] | None = None):
    """Load the shared config with volume paths applied."""
    from fvqa.config import load_config
    from fvqa.utils import setup_logging

    setup_logging()
    return load_config(overrides=VOLUME_OVERRIDES + list(extra_overrides or []))


def _login() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token)
        print("✓ logged in to HuggingFace")


def _ensure_fvqa_downloaded(root: str) -> None:
    """Download and extract the FVQA release into `root` if not already there.

    FVQA is not on HuggingFace, so there is no `datasets.load_dataset` to
    lean on — a plain HTTP download + zip extraction, done once per volume.
    """
    import subprocess
    import zipfile

    marker = os.path.join(root, "Name_Lists")
    if os.path.isdir(marker):
        print(f"✓ FVQA already present at {root}")
        return

    os.makedirs(root, exist_ok=True)
    zip_path = "/tmp/fvqa.zip"
    print(f"downloading FVQA release (~451MB) from {FVQA_ZIP_URL}")
    subprocess.run(["wget", "-q", "-O", zip_path, FVQA_ZIP_URL], check=True)

    print("extracting...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    os.remove(zip_path)
    print(f"✓ extracted to {root}")


@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    timeout=3600,
    memory=8192,
)
def prepare_dataset(limit: int | None = None, grounding: bool = False):
    """Download FVQA (if needed) and write train/val/test splits to the volume."""
    from fvqa.data.fvqa import prepare_fvqa

    config = _load([f"data.grounding.enabled={str(grounding).lower()}"])

    _ensure_fvqa_downloaded(config.data.root)
    counts = prepare_fvqa(config, limit=limit)
    data_volume.commit()

    print(f"\n✓ prepared: {counts}")
    return counts


@app.function(
    image=image,
    gpu="A100",
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,
    memory=32768,
)
def train_model(
    num_epochs: int | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    resume: bool = True,
    dry_run: bool = False,
):
    """Fine-tune Qwen3-VL with LoRA.

    Arguments left as None fall back to config/config.yaml.
    """
    from fvqa.train.runner import run_training

    _login()

    overrides = []
    if num_epochs is not None:
        overrides.append(f"training.num_train_epochs={num_epochs}")
    if batch_size is not None:
        overrides.append(f"training.per_device_train_batch_size={batch_size}")
    if grad_accum is not None:
        overrides.append(f"training.gradient_accumulation_steps={grad_accum}")

    config = _load(overrides)

    print(f"model:  {config.model.model_id}")
    print(f"epochs: {config.training.num_train_epochs}")
    print(f"batch:  {config.training.effective_batch_size} effective")
    print(f"LoRA:   rank={config.model.lora.rank} alpha={config.model.lora.alpha}")

    exit_code = run_training(
        config,
        trainer_dir="/root/Qwen-VL-Series-Finetune",
        num_gpus=1,
        resume=resume,
        dry_run=dry_run,
    )
    checkpoint_volume.commit()

    output_dir = config.training.output_dir
    checkpoints = sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []
    return {"exit_code": exit_code, "checkpoints": checkpoints}


@app.function(image=image, gpu="A100", volumes=VOLUMES, timeout=7200, memory=32768)
def evaluate_model(
    model_path: str | None = None,
    checkpoint: str | None = None,
    num_samples: int | None = None,
    split: str = "val",
):
    """Score a model on a split.

    Args:
        model_path: A HuggingFace model id or absolute path. Use this to
            score the un-finetuned base model — the baseline a fine-tuning
            result has to be read against.
        checkpoint: A checkpoint directory name under the output dir.
        num_samples: How many samples to score; -1 for the whole split.
        split: Which split to score.

    With neither model_path nor checkpoint, the newest checkpoint is used.
    """
    from fvqa.evaluation.runner import evaluate, format_scores
    from fvqa.model import VQAModel
    from fvqa.train.command import resolve_model_source

    config = _load()
    path = resolve_model_source(
        model_path=model_path,
        checkpoint=checkpoint,
        output_dir=config.training.output_dir,
    )
    print(f"📂 evaluating {path}")

    model = VQAModel.from_pretrained(path, config)
    result = evaluate(
        model,
        config,
        split=split,
        num_samples=num_samples,
        output_path=f"{CHECKPOINT_DIR}/eval_{os.path.basename(path)}_{split}.json",
    )
    checkpoint_volume.commit()

    print(format_scores(result))
    return {"checkpoint": os.path.basename(path), **result["metrics"]}


@app.function(image=image, volumes=VOLUMES)
def check_status():
    """Report what is on the volumes."""
    import json

    config = _load()
    status: dict[str, object] = {}

    for split in ("train", "val", "test"):
        path = config.data.split_file(split)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                status[split] = len(json.load(handle))

    if os.path.isdir(config.data.image_folder):
        status["images"] = len(os.listdir(config.data.image_folder))

    output_dir = config.training.output_dir
    if os.path.isdir(output_dir):
        status["checkpoints"] = sorted(
            entry for entry in os.listdir(output_dir) if entry.startswith("checkpoint-")
        )

    print(json.dumps(status, indent=2))
    return status


@app.local_entrypoint()
def main(
    step: str = "all",
    num_epochs: int = 0,
    num_eval: int = 200,
    grounding: bool = False,
    limit: int = 0,
    baseline: bool = False,
):
    """Run the pipeline.

    Args:
        step: prepare | train | evaluate | baseline | all
        num_epochs: 0 to use the value in config.yaml
        num_eval: samples to score; -1 for the whole split
        grounding: prepare the data with oracle-fact grounding enabled
        limit: process at most this many questions; 0 for all
        baseline: score the un-finetuned base model instead of a checkpoint

    Scoring the base model needs no training at all:
        modal run scripts/train_on_modal.py --step baseline
    """
    valid = {"prepare", "train", "evaluate", "baseline", "all"}
    if step not in valid:
        raise SystemExit(f"--step must be one of {sorted(valid)}, got {step!r}")

    if step in ("prepare", "all"):
        print("\n📦 preparing dataset")
        print(prepare_dataset.remote(limit=limit or None, grounding=grounding))

    if step in ("train", "all"):
        print("\n🎯 training")
        print(train_model.remote(num_epochs=num_epochs or None))

    if step == "baseline" or baseline:
        print("\n📊 scoring the base model (no fine-tuning)")
        config_model_id = "Qwen/Qwen3-VL-8B-Instruct"
        print(evaluate_model.remote(model_path=config_model_id, num_samples=num_eval))
    elif step in ("evaluate", "all"):
        print("\n📊 evaluating")
        print(evaluate_model.remote(num_samples=num_eval))
