"""Vi-VQA training pipeline on Modal.

All dataset, configuration and command-building logic lives in the
`vivqa` package and is shared with the local run — this file contains
only what is genuinely Modal-specific: the container image, the volumes
and the function boundaries. The previous version reimplemented dataset
preparation here, and the two copies had already drifted apart.

Usage:
    pip install modal
    modal token new
    modal secret create huggingface-secret HF_TOKEN=hf_...

    modal run scripts/train_on_modal.py --step prepare
    modal run scripts/train_on_modal.py --step train
    modal run scripts/train_on_modal.py --step evaluate
    modal run scripts/train_on_modal.py::check_status

Estimated training time for 2 epochs over ~31k QA pairs:
    A10G ~20-25h · A100 ~8-12h · H100 ~4-6h
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("vi-vqa-training")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        # Qwen3VLForConditionalGeneration does not exist before 4.57.
        "transformers>=4.57.0",
        "accelerate>=0.34.0",
        "peft>=0.13.0",
        "bitsandbytes>=0.44.0",
        "deepspeed>=0.15.0",
        "datasets>=3.0.0",
        "pillow>=10.0.0",
        "qwen-vl-utils>=0.0.8",
        "huggingface_hub>=0.26.0",
        "pyyaml>=6.0",
        "tqdm",
        "tensorboard",
    )
    .env({"CUDA_HOME": "/usr/local/cuda", "VIVQA_CONFIG": "/root/config/config.yaml"})
    .run_commands("pip install flash-attn --no-build-isolation || echo 'flash-attn unavailable'")
    # Ship the package and config instead of duplicating their logic here.
    .add_local_dir(REPO_ROOT / "src" / "vivqa", remote_path="/root/vivqa")
    .add_local_dir(REPO_ROOT / "config", remote_path="/root/config")
)

data_volume = modal.Volume.from_name("vi-vqa-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("vi-vqa-checkpoints", create_if_missing=True)

DATA_DIR = "/data"
CHECKPOINT_DIR = "/checkpoints"
VOLUMES = {DATA_DIR: data_volume, CHECKPOINT_DIR: checkpoint_volume}

# Point the shared config at the mounted volumes. Everything else —
# learning rates, LoRA rank, grounding — still comes from config.yaml.
VOLUME_OVERRIDES = [
    f"data.data_dir={DATA_DIR}",
    f"data.image_folder={DATA_DIR}/images",
    f"training.output_dir={CHECKPOINT_DIR}/qwen3vl-vivqa",
]


def _load(extra_overrides: list[str] | None = None):
    """Load the shared config with volume paths applied."""
    from vivqa.config import load_config
    from vivqa.utils import setup_logging

    setup_logging()
    return load_config(overrides=VOLUME_OVERRIDES + list(extra_overrides or []))


def _login() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token)
        print("✓ logged in to HuggingFace")


@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=8192,
)
def prepare_dataset(
    limit: int | None = None,
    grounding: bool = False,
    streaming: bool = False,
):
    """Download the dataset and write train/val/test splits to the volume.

    `streaming=True` with a `limit` fetches only the records it needs
    instead of pulling the whole split first.
    """
    from vivqa.data.prepare import prepare

    _login()
    config = _load([
        f"data.grounding.enabled={str(grounding).lower()}",
        f"data.streaming={str(streaming).lower()}",
    ])

    counts = prepare(config, limit=limit)
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
    from vivqa.train.runner import run_training

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
    from vivqa.evaluation.runner import evaluate, format_scores
    from vivqa.model import VQAModel
    from vivqa.train.command import resolve_model_source

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
    streaming: bool = False,
    limit: int = 0,
    baseline: bool = False,
):
    """Run the pipeline.

    Args:
        step: prepare | train | evaluate | baseline | all
        num_epochs: 0 to use the value in config.yaml
        num_eval: samples to score; -1 for the whole split
        grounding: prepare the data with description grounding enabled
        streaming: stream the dataset instead of downloading the split
        limit: process at most this many records; 0 for all
        baseline: score the un-finetuned base model instead of a checkpoint

    Scoring the base model needs no training at all:
        modal run scripts/train_on_modal.py --step baseline
    """
    valid = {"prepare", "train", "evaluate", "baseline", "all"}
    if step not in valid:
        raise SystemExit(f"--step must be one of {sorted(valid)}, got {step!r}")

    if step in ("prepare", "all"):
        print("\n📦 preparing dataset")
        print(prepare_dataset.remote(
            limit=limit or None, grounding=grounding, streaming=streaming
        ))

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
