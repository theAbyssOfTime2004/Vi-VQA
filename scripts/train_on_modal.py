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
        # Needed by the trainer, not by fvqa, and not obviously so:
        # src/dataset/__init__.py imports the DPO and classification
        # datasets alongside the SFT one, so an SFT run still has to
        # import dpo_dataset -> ujson and dpo_dataset -> params -> trl.
        # Both were missing until a real run hit them one at a time;
        # smoke level 1 now checks the whole import graph instead. Floors
        # match the trainer's own requirements.txt at the pinned commit.
        "trl>=0.25.0",
        "ujson>=5.10.0",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "FVQA_CONFIG": "/root/config/config.yaml",
            # Explicit rather than relying on the working directory
            # happening to be importable: `import fvqa` should not depend
            # on where Modal chose to start the process.
            "PYTHONPATH": "/root/src",
        }
    )
    .run_commands("pip install flash-attn --no-build-isolation || echo 'flash-attn unavailable'")
    # Ship the package and config instead of duplicating their logic here.
    .add_local_dir(REPO_ROOT / "src", remote_path="/root/src")
    .add_local_dir(REPO_ROOT / "config", remote_path="/root/config")
    .add_local_dir(REPO_ROOT / "scripts", remote_path="/root/scripts")
)

#: Loading a model needs the HuggingFace token. Every function that loads
#: one gets this — including the baseline and the smoke tests, which used
#: to be the ones that quietly failed on a gated repo.
HF_SECRET = modal.Secret.from_name("huggingface-secret")

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
    # On the volume so vision seeds survive the container: recomputing
    # them is a VLM call per question, and they do not depend on any
    # retrieval setting an experiment might vary.
    f"retrieval.seed_cache_dir={DATA_DIR}/vision-seeds",
]


def _load(extra_overrides: list[str] | None = None):
    """Load the shared config with volume paths applied.

    A `ConfigError` is re-raised as a plain `RuntimeError` because the
    caller is on another machine. Modal pickles the remote exception and
    unpickles it locally, and `fvqa` is not installed there — so a
    `ConfigError` arrives as "ModuleNotFoundError: No module named 'fvqa'"
    wrapped around the real message, which buries what actually went
    wrong. RuntimeError is in the standard library, so it survives the
    trip intact.
    """
    from fvqa.config import ConfigError, load_config
    from fvqa.utils import setup_logging

    setup_logging()
    try:
        return load_config(overrides=VOLUME_OVERRIDES + list(extra_overrides or []))
    except ConfigError as error:
        raise RuntimeError(f"config error: {error}") from None


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
    print(f"tuning: {config.model.tuning_method}")
    if config.model.uses_lora:
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


@app.function(
    image=image,
    gpu="A100",
    volumes=VOLUMES,
    # It loads a model, so it needs the token just as much as training
    # does. Without this a gated or rate-limited repo fails here only.
    secrets=[HF_SECRET],
    timeout=7200,
    memory=32768,
)
def evaluate_model(
    model_path: str | None = None,
    checkpoint: str | None = None,
    num_samples: int | None = None,
    split: str = "val",
    condition: str = "stored",
    base_model: str | None = None,
):
    """Score a model on a split.

    Args:
        model_path: A HuggingFace model id or absolute path. Use this to
            score the un-finetuned base model — the baseline a fine-tuning
            result has to be read against.
        checkpoint: A checkpoint directory name under the output dir.
        num_samples: How many samples to score; -1 for the whole split.
        split: Which split to score.
        condition: Prompt condition — stored / no-context / style /
            oracle-fact / oracle-seed-graph / vision-seed-graph.
        base_model: Base model for a LoRA adapter whose recorded
            base_model_name_or_path no longer resolves.

    With neither model_path nor checkpoint, the newest checkpoint is used.
    """
    from fvqa.evaluation.runner import evaluate, format_scores
    from fvqa.model import VQAModel
    from fvqa.train.command import resolve_model_source

    _login()

    config = _load()
    path = resolve_model_source(
        model_path=model_path,
        checkpoint=checkpoint,
        output_dir=config.training.output_dir,
    )
    print(f"📂 evaluating {path} (condition: {condition})")

    model = VQAModel.from_pretrained(path, config, base_model_id=base_model)
    result = evaluate(
        model,
        config,
        split=split,
        num_samples=num_samples,
        condition=condition,
        output_path=(
            f"{CHECKPOINT_DIR}/eval_{os.path.basename(path)}_{split}_{condition}.json"
        ),
    )
    checkpoint_volume.commit()
    # Vision seeding writes into the data volume; without this commit the
    # cache is thrown away with the container and the next run pays for
    # the same VLM calls again.
    data_volume.commit()

    print(format_scores(result))
    return {"checkpoint": os.path.basename(path), "condition": condition, **result["metrics"]}



# --------------------------------------------------------------------------
# Smoke tests
#
# Four levels, cheapest first, each with its own short timeout rather than
# borrowing the 24-hour training function's. A smoke test that can hang
# for a day is not a smoke test — the point is to find out quickly, and a
# level that fails tells you where without having paid for the levels
# above it.
# --------------------------------------------------------------------------


@app.function(image=image, timeout=600)
def smoke_import():
    """Level 1 — the image is built correctly. No GPU, no data, no model.

    Catches the failures that need nothing else present: a package that
    did not ship, a PYTHONPATH that does not include it, a config that
    does not parse, a graph module that cannot be imported.
    """
    import sys

    print(f"python: {sys.version.split()[0]}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")

    import fvqa
    from fvqa.data.fvqa_graph import KnowledgeGraph
    from fvqa.evaluation.conditions import CONDITIONS
    from fvqa.retrieval import GraphRetriever

    print(f"fvqa imported from {fvqa.__file__}")

    config = _load()
    print(f"config: project={config.project_name} model={config.model.model_id}")
    print(f"conditions: {list(CONDITIONS)}")

    # A three-triple graph: proves the traversal code runs here, without
    # waiting on the 451MB download the next level does.
    graph = KnowledgeGraph(
        {
            "f1": {
                "KB": "conceptnet", "e1": "trumpet", "e2": "jazz club",
                "e1_label": "trumpet", "e2_label": "jazz club",
                "r": "/r/AtLocation", "surface": "[[trumpet]] in [[jazz club]]",
            }
        }
    )
    retriever = GraphRetriever(graph)
    result = retriever.retrieve(["trumpet"], "Where?", max_hops=1)
    assert result.status == "ok", result.status
    print(f"mini-graph retrieval: {result.status}, {len(result.facts)} fact(s)")

    # The trainer's dependencies are part of "is the image built
    # correctly", and checking them here costs a small git fetch and no
    # GPU. Two packages (trl, ujson) were missing from the image and each
    # surfaced separately, three minutes into a GPU container, because
    # nothing checked until training actually reached the import.
    trainer = _check_trainer_imports(config)
    flags = _check_trainer_flags(trainer["repo"])

    return {
        "level": 1,
        "ok": True,
        "conditions": list(CONDITIONS),
        "trainer": trainer,
        "flags_checked": flags,
    }


def _check_trainer_flags(repo: str) -> int:
    """Verify every flag the training command can emit is one the trainer takes.

    Worth running here rather than only by hand: transformers is
    installed in this image, so the inherited Trainer fields are read
    from the real thing. That is the difference between catching
    `--warmup_ratio` (removed in transformers 5) in seconds with no GPU,
    and catching it four minutes into an A100 container.
    """
    import sys

    sys.path.insert(0, "/root/scripts")
    from pathlib import Path

    from check_trainer_flags import _emitted_flags, _trainer_field_names  # type: ignore

    known, exact = _trainer_field_names(Path(repo) / "src")
    emitted = _emitted_flags()
    unknown = sorted(emitted - known)

    if unknown:
        raise RuntimeError(
            f"the training command emits {unknown}, which the trainer's parser does "
            "not accept. Fix src/fvqa/train/command.py or the config field behind it."
        )

    print(
        f"trainer flags: all {len(emitted)} accepted "
        f"({'exact' if exact else 'partial — transformers missing'})"
    )
    return len(emitted)


def _check_trainer_imports(config) -> dict[str, object]:
    """Verify every package the trainer's entry point imports is installed."""
    import sys

    sys.path.insert(0, "/root/scripts")
    from check_trainer_imports import third_party_imports  # type: ignore

    from fvqa.train.runner import ensure_trainer_repo

    repo = ensure_trainer_repo(
        "/root/Qwen-VL-Series-Finetune",
        url=config.trainer.repo_url,
        revision=config.trainer.revision,
    )

    import importlib.util
    from pathlib import Path

    reached = third_party_imports(Path(repo) / "src", Path("train") / "train_sft.py")
    missing = {
        name: chain
        for name, chain in reached.items()
        if importlib.util.find_spec(name) is None
    }

    print(f"trainer imports: {len(reached)} package(s) reachable from train_sft.py")
    if missing:
        for name, chain in sorted(missing.items()):
            print(f"  MISSING {name} — reached via {' -> '.join(chain)}")
        raise RuntimeError(
            f"the image is missing {sorted(missing)}, which the trainer imports. "
            "Add them to the pip_install list in scripts/train_on_modal.py."
        )

    print(f"  all {len(reached)} present")
    return {
        "checked": sorted(reached),
        "revision": config.trainer.revision[:12],
        "repo": repo,
    }


@app.function(image=image, volumes={DATA_DIR: data_volume}, timeout=1800, memory=8192)
def smoke_prepare(limit: int = 40):
    """Level 2 — the dataset downloads, parses and splits. Still no GPU.

    `--limit` keeps this quick, which means the val split can come out
    tiny or empty: splitting happens per *image*, so a small slice of
    questions may put every image on one side. That is expected here —
    this level checks the files are written and well-formed, and is not a
    place to read anything into the split sizes.
    """
    import json

    from fvqa.data.fvqa import prepare_fvqa

    config = _load()
    _ensure_fvqa_downloaded(config.data.root)
    counts = prepare_fvqa(config, limit=limit)
    data_volume.commit()

    written = {}
    for split in ("train", "val", "test"):
        path = config.data.split_file(split)
        if not os.path.exists(path):
            written[split] = None
            continue
        with open(path, encoding="utf-8") as handle:
            samples = json.load(handle)
        written[split] = len(samples)
        if samples:
            sample = samples[0]
            missing = [
                key for key in ("id", "image", "conversations", "fvqa_question")
                if key not in sample
            ]
            assert not missing, f"{split}.json sample is missing {missing}"

    print(f"counts: {counts}")
    print(f"written: {written}")
    return {"level": 2, "ok": True, "counts": counts, "written": written}


@app.function(
    image=image,
    gpu="A100",
    volumes=VOLUMES,
    secrets=[HF_SECRET],
    timeout=1800,
    memory=32768,
)
def smoke_baseline(num_samples: int = 3, condition: str = "no-context"):
    """Level 3 — the base model loads on a GPU and generates. No training.

    The first level that costs GPU time, and the last one that can fail
    for reasons unrelated to training: weights that will not download,
    an attention implementation that is not available, a processor that
    rejects the images.
    """
    from fvqa.evaluation.runner import evaluate, format_scores
    from fvqa.model import VQAModel

    _login()
    config = _load()

    model = VQAModel.from_pretrained(config.model.model_id, config)
    result = evaluate(
        model, config, split="train", num_samples=num_samples, condition=condition
    )
    data_volume.commit()

    print(format_scores(result))
    assert result["num_samples"] > 0, "no sample generated successfully"
    return {
        "level": 3,
        "ok": True,
        "condition": condition,
        "scored": result["num_samples"],
        "failed": result["num_failed"],
        "sample_prediction": result["predictions"][0]["prediction"] if result["predictions"] else None,
    }


@app.function(
    image=image,
    gpu="A100",
    volumes=VOLUMES,
    secrets=[HF_SECRET],
    timeout=3600,
    memory=32768,
)
def smoke_train(max_steps: int = 2):
    """Level 4 — two optimizer steps, then reload what they wrote.

    The reload is the half that matters. Training writing *a* directory
    proves little; the failure this catches is an adapter checkpoint that
    cannot be loaded back — which is invisible until someone tries to
    evaluate a real run days later.
    """
    from fvqa.checkpoint import inspect_checkpoint
    from fvqa.evaluation.runner import evaluate
    from fvqa.model import VQAModel
    from fvqa.train.command import latest_checkpoint
    from fvqa.train.runner import run_training

    _login()

    output_dir = f"{CHECKPOINT_DIR}/smoke"
    config = _load(
        [
            f"training.output_dir={output_dir}",
            f"training.max_steps={max_steps}",
            "training.save_steps=1",
            "training.eval_strategy=no",
            "training.load_best_model_at_end=false",
            "training.logging_steps=1",
        ]
    )

    exit_code = run_training(
        config, trainer_dir="/root/Qwen-VL-Series-Finetune", num_gpus=1, resume=False
    )
    checkpoint_volume.commit()
    assert exit_code == 0, f"training exited with {exit_code}"

    checkpoint = latest_checkpoint(output_dir)
    assert checkpoint, f"training wrote no checkpoint under {output_dir}"

    info = inspect_checkpoint(checkpoint)
    print(f"checkpoint {checkpoint} detected as: {info.kind}")
    print(f"  base model:       {info.base_model_id}")
    print(f"  non-LoRA weights: {info.non_lora_path}")

    model = VQAModel.from_pretrained(checkpoint, config)
    result = evaluate(model, config, split="train", num_samples=1, condition="stored")
    assert result["num_samples"] > 0, "reloaded checkpoint generated nothing"

    return {
        "level": 4,
        "ok": True,
        "checkpoint": os.path.basename(checkpoint),
        "kind": info.kind,
        "base_model": info.base_model_id,
        "has_non_lora_weights": info.non_lora_path is not None,
        "prediction": result["predictions"][0]["prediction"],
    }


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
    condition: str = "stored",
    smoke_level: int = 4,
):
    """Run the pipeline.

    Args:
        step: smoke | prepare | train | evaluate | baseline | all
        num_epochs: 0 to use the value in config.yaml
        num_eval: samples to score; -1 for the whole split
        grounding: prepare the data with oracle-fact grounding enabled
        limit: process at most this many questions; 0 for all
        baseline: score the un-finetuned base model instead of a checkpoint
        condition: prompt condition for evaluation
        smoke_level: highest smoke level to run (1-4)

    Scoring the base model needs no training at all:
        modal run scripts/train_on_modal.py --step baseline
    """
    valid = {"smoke", "prepare", "train", "evaluate", "baseline", "all"}
    if step not in valid:
        raise SystemExit(f"--step must be one of {sorted(valid)}, got {step!r}")

    if step == "smoke":
        # Cheapest first, and stop at the first failure: a level that
        # fails makes every level above it uninformative.
        levels = [
            (1, "import + config + mini-graph (no GPU)", smoke_import),
            (2, "download + prepare (no GPU)", smoke_prepare),
            (3, "base model loads and generates (GPU)", smoke_baseline),
            (4, "2 training steps + adapter reload (GPU)", smoke_train),
        ]
        if not 1 <= smoke_level <= len(levels):
            # Falling through with nothing to run would print the success
            # line having tested nothing at all.
            raise SystemExit(
                f"--smoke-level must be between 1 and {len(levels)}, got {smoke_level}"
            )

        for number, description, function in levels:
            if number > smoke_level:
                break
            print(f"\n🔥 smoke {number}/{smoke_level}: {description}")
            # smoke_prepare is the only level that takes the shared
            # --limit; the others have their own sensible smoke defaults.
            if function is smoke_prepare and limit:
                print(function.remote(limit=limit))
            else:
                print(function.remote())
        print(f"\n✅ smoke levels 1-{smoke_level} passed")
        return

    if step in ("prepare", "all"):
        print("\n📦 preparing dataset")
        print(prepare_dataset.remote(limit=limit or None, grounding=grounding))

    if step in ("train", "all"):
        print("\n🎯 training")
        print(train_model.remote(num_epochs=num_epochs or None))

    if step == "baseline" or baseline:
        print("\n📊 scoring the base model (no fine-tuning)")
        config_model_id = "Qwen/Qwen3-VL-8B-Instruct"
        print(
            evaluate_model.remote(
                model_path=config_model_id, num_samples=num_eval, condition=condition
            )
        )
    elif step in ("evaluate", "all"):
        print("\n📊 evaluating")
        print(evaluate_model.remote(num_samples=num_eval, condition=condition))
