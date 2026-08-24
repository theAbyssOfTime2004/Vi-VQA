"""Typed configuration for FVQA.

`config/config.yaml` is the only place hyperparameters live. Every entry
point — the CLI, the local training script and the Modal pipeline — goes
through :func:`load_config`, so a value can never drift between them.

Unknown keys are an error rather than a warning: a typo in a YAML key is
otherwise invisible until a 20-hour training run finishes with the wrong
learning rate.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

__all__ = [
    "Config",
    "ConfigError",
    "DataConfig",
    "EvaluationConfig",
    "GroundingConfig",
    "InferenceConfig",
    "LoraConfig",
    "ModelConfig",
    "QuantizationConfig",
    "RetrievalConfig",
    "SplitConfig",
    "TrainerConfig",
    "TrainingConfig",
    "default_config_path",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when the configuration file is malformed."""


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


@dataclass
class SplitConfig:
    train: float = 0.90
    val: float = 0.05
    test: float = 0.05

    def validate(self, path: str) -> None:
        for name in ("train", "val", "test"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{path}.{name} must be in [0, 1], got {value}")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"{path} fractions must sum to 1.0, got {total:.6f} "
                f"(train={self.train}, val={self.val}, test={self.test})"
            )
        if self.train <= 0:
            raise ConfigError(f"{path}.train must be greater than 0")

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass
class GroundingConfig:
    """Knowledge grounding using the dataset's own `description` field."""

    enabled: bool = False
    mode: str = "prefix"
    max_chars: int = 1200

    # Used in 'prefix' mode: context and question folded into one user turn.
    template: str = "Fact: {description}\n\nUsing the fact above, answer: {question}"
    # Used in 'system' mode: context only. The question stays in the user
    # turn, so this template must not repeat it.
    system_template: str = "Fact: {description}"

    VALID_MODES = ("prefix", "system")

    def validate(self, path: str) -> None:
        if self.mode not in self.VALID_MODES:
            raise ConfigError(
                f"{path}.mode must be one of {self.VALID_MODES}, got {self.mode!r}"
            )
        if self.max_chars < 1:
            raise ConfigError(f"{path}.max_chars must be positive, got {self.max_chars}")

        if "{description}" not in self.template:
            raise ConfigError(f"{path}.template must contain a {{description}} placeholder")
        if "{question}" not in self.template:
            raise ConfigError(f"{path}.template must contain a {{question}} placeholder")

        if "{description}" not in self.system_template:
            raise ConfigError(
                f"{path}.system_template must contain a {{description}} placeholder"
            )
        if "{question}" in self.system_template:
            raise ConfigError(
                f"{path}.system_template must not contain {{question}}: in 'system' mode "
                "the question stays in the user turn and would otherwise be duplicated"
            )


@dataclass
class RetrievalConfig:
    """Graph retrieval: how far to walk, and how much to keep.

    Separate from `data` on purpose. `max_hops` used to live there, but
    it describes the retrieval algorithm, not the dataset — FVQA's graph
    is the same graph whether you walk one hop or three. Anything that
    changes a score without changing the data belongs here, and gets
    written into the result file so two runs are distinguishable.
    """

    enabled: bool = False
    max_hops: int = 2
    #: Cap on graph nodes the seed guesses resolve to.
    max_seed_entities: int = 5
    #: Cap on facts the traversal collects before ranking. A
    #: well-connected seed reaches a large slice of the graph well before
    #: max_hops runs out. 300 is measured, not guessed: on 981 real
    #: questions with an oracle seed at 1 hop, recall@5 of the supporting
    #: fact runs 58.0% at cap 50, 59.4% at 100, 61.1% at 300, 62.7% at
    #: 5000 — and 300 costs 1.6 ms/question, which is nothing beside a
    #: single VLM generate call.
    max_candidate_facts: int = 300
    #: How many ranked facts go into the prompt.
    top_k_facts: int = 5
    ranking_method: str = "lexical"

    VALID_RANKING_METHODS = ("lexical",)

    def validate(self, path: str) -> None:
        if self.max_hops < 1:
            raise ConfigError(f"{path}.max_hops must be positive, got {self.max_hops}")
        for name in ("max_seed_entities", "max_candidate_facts", "top_k_facts"):
            value = getattr(self, name)
            if value < 1:
                raise ConfigError(f"{path}.{name} must be positive, got {value}")
        if self.top_k_facts > self.max_candidate_facts:
            raise ConfigError(
                f"{path}.top_k_facts ({self.top_k_facts}) cannot exceed "
                f"max_candidate_facts ({self.max_candidate_facts}): ranking cannot "
                "return more facts than the traversal collected"
            )
        if self.ranking_method not in self.VALID_RANKING_METHODS:
            raise ConfigError(
                f"{path}.ranking_method must be one of {self.VALID_RANKING_METHODS}, "
                f"got {self.ranking_method!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        """The provenance block written into a result file."""
        return {
            "enabled": self.enabled,
            "max_hops": self.max_hops,
            "max_seed_entities": self.max_seed_entities,
            "max_candidate_facts": self.max_candidate_facts,
            "top_k_facts": self.top_k_facts,
            "ranking_method": self.ranking_method,
        }


@dataclass
class DataConfig:
    """Where the local FVQA release lives, and how to traverse its graph.

    FVQA has to be downloaded and extracted by hand (no `datasets.load_dataset`
    support): https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=1
    `root` should point at the directory that zip extracts into, containing
    `Name_Lists/` and `new_dataset_release/`. `image_folder` defaults to
    where images live inside that layout — override both together if you
    extract to a different `root`.
    """

    data_dir: str = "./data"
    root: str = "./data/fvqa"
    image_folder: str = "./data/fvqa/new_dataset_release/images"
    fold: int = 0  # FVQA ships 5 official train/test folds, numbered 0-4

    splits: SplitConfig = field(default_factory=SplitConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)

    def validate(self, path: str) -> None:
        if not 0 <= self.fold <= 4:
            raise ConfigError(f"{path}.fold must be in [0, 4], got {self.fold}")
        self.splits.validate(f"{path}.splits")
        self.grounding.validate(f"{path}.grounding")

    def split_file(self, split: str) -> str:
        """Path of the JSON file holding a given split."""
        return os.path.join(self.data_dir, f"{split}.json")


@dataclass
class LoraConfig:
    """Adapter shape. Read only when `tuning_method` is lora or qlora."""

    rank: int = 128
    alpha: int = 256
    dropout: float = 0.05
    num_lora_modules: int = -1
    namespan_exclude: list[str] = field(
        default_factory=lambda: ["lm_head", "embed_tokens"]
    )
    vision_lora: bool = False

    def validate(self, path: str) -> None:
        if self.rank < 1:
            raise ConfigError(f"{path}.rank must be positive, got {self.rank}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"{path}.dropout must be in [0, 1), got {self.dropout}")


@dataclass
class QuantizationConfig:
    """4-bit base weights. Read only when `tuning_method` is qlora.

    These are the two knobs the trainer actually exposes (`--quant_type`,
    `--double_quant`). The compute dtype is deliberately absent: the
    trainer derives it from `training.bf16`/`training.fp16` and offers no
    flag for it, so a field here could only ever be decorative.
    """

    quant_type: str = "nf4"
    double_quant: bool = True

    def validate(self, path: str) -> None:
        valid = ("nf4", "fp4")
        if self.quant_type not in valid:
            raise ConfigError(
                f"{path}.quant_type must be one of {valid}, got {self.quant_type!r}"
            )


@dataclass
class ModelConfig:
    """The model, and how it is to be tuned.

    `tuning_method` is a single enum rather than a pair of `lora.enabled`
    / `qlora.enabled` booleans on purpose. Two booleans describe four
    states, only three of which mean anything, and the fourth
    (`lora=false, qlora=true`) is not merely redundant — it is what QLoRA
    was previously configured as, and it silently produced a run that
    trained nothing: 4-bit frozen base weights with no adapters attached.
    QLoRA *is* LoRA adapters on a quantized base; with an enum the
    contradictory state cannot be written down.
    """

    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "bfloat16"
    use_flash_attn: bool = True
    image_min_pixels: int = 256 * 32 * 32
    image_max_pixels: int = 1280 * 32 * 32
    tuning_method: str = "lora"
    lora: LoraConfig = field(default_factory=LoraConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    VALID_TUNING_METHODS = ("full", "lora", "qlora")

    @property
    def uses_lora(self) -> bool:
        """QLoRA trains adapters too — that is the whole point of it."""
        return self.tuning_method in ("lora", "qlora")

    @property
    def uses_quantization(self) -> bool:
        return self.tuning_method == "qlora"

    def validate(self, path: str) -> None:
        valid_dtypes = ("bfloat16", "float16", "float32", "auto")
        if self.torch_dtype not in valid_dtypes:
            raise ConfigError(
                f"{path}.torch_dtype must be one of {valid_dtypes}, got {self.torch_dtype!r}"
            )
        if self.image_min_pixels >= self.image_max_pixels:
            raise ConfigError(
                f"{path}.image_min_pixels ({self.image_min_pixels}) must be smaller "
                f"than image_max_pixels ({self.image_max_pixels})"
            )
        if self.tuning_method not in self.VALID_TUNING_METHODS:
            raise ConfigError(
                f"{path}.tuning_method must be one of {self.VALID_TUNING_METHODS}, "
                f"got {self.tuning_method!r}"
            )
        self.lora.validate(f"{path}.lora")
        self.quantization.validate(f"{path}.quantization")


@dataclass
class TrainerConfig:
    """Which commit of the external trainer repo to run against.

    `2U1/Qwen-VL-Series-Finetune` moves independently of this project —
    an upstream flag rename or restructure can silently break
    `build_train_command`'s output. Pinning a revision here means "config
    + code + this commit" is the whole reproducibility story for a run,
    instead of "whatever HEAD happened to be that day".
    """

    repo_url: str = "https://github.com/2U1/Qwen-VL-Series-Finetune.git"
    revision: str = "70c7b2fcb0e276b1fa4b136852e9b862ce8730fa"

    def validate(self, path: str) -> None:
        if not self.repo_url.strip():
            raise ConfigError(f"{path}.repo_url must not be empty")
        if not self.revision.strip():
            raise ConfigError(f"{path}.revision must not be empty")


@dataclass
class TrainingConfig:
    output_dir: str = "./checkpoints/qwen3vl-fvqa"
    num_train_epochs: int = 2
    # Stop after this many optimizer steps regardless of epochs. None
    # runs the full schedule; a small value is what makes a GPU smoke
    # test possible — one that proves the model loads, a batch reaches
    # the forward pass, gradients flow and a checkpoint is written,
    # without paying for a real run to find out.
    max_steps: int | None = None
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    learning_rate: float = 2e-5
    vision_lr: float = 2e-6
    merger_lr: float = 2e-5
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    optim: str = "adamw_bnb_8bit"
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    lazy_preprocess: bool = True

    freeze_vision_tower: bool = True
    freeze_llm: bool = False
    freeze_merger: bool = False

    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 500
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    logging_steps: int = 50
    report_to: str = "tensorboard"
    deepspeed: str | None = "scripts/zero2.json"

    def validate(self, path: str) -> None:
        if self.bf16 and self.fp16:
            raise ConfigError(f"{path}: bf16 and fp16 are mutually exclusive")
        if self.num_train_epochs <= 0:
            raise ConfigError(
                f"{path}.num_train_epochs must be positive, got {self.num_train_epochs}"
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ConfigError(
                f"{path}.max_steps must be positive when set, got {self.max_steps}"
            )
        for name in ("per_device_train_batch_size", "gradient_accumulation_steps"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{path}.{name} must be at least 1")
        if self.freeze_llm and self.freeze_vision_tower and self.freeze_merger:
            raise ConfigError(f"{path}: every component is frozen, nothing would train")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ConfigError(
                f"{path}.warmup_ratio must be in [0, 1], got {self.warmup_ratio}"
            )

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass
class InferenceConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # A system turn that shapes *how* the model answers, carrying no
    # information about the image. Kept separate from data.grounding on
    # purpose: grounding hands the model facts, this only sets style, and
    # mixing them makes it impossible to tell which one moved a score.
    system_prompt: str = ""

    def validate(self, path: str) -> None:
        if self.max_new_tokens < 1:
            raise ConfigError(
                f"{path}.max_new_tokens must be positive, got {self.max_new_tokens}"
            )
        if self.temperature < 0:
            raise ConfigError(f"{path}.temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ConfigError(f"{path}.top_p must be in (0, 1], got {self.top_p}")


@dataclass
class EvaluationConfig:
    split: str = "val"
    num_samples: int = 200
    temperature: float = 0.0
    metrics: list[str] = field(
        default_factory=lambda: ["exact_match", "similarity", "bleu", "rouge_l", "cider"]
    )

    def validate(self, path: str) -> None:
        if self.num_samples == 0 or self.num_samples < -1:
            raise ConfigError(
                f"{path}.num_samples must be positive or -1 (all), got {self.num_samples}"
            )
        # Imported lazily to keep the dependency direction one-way.
        from fvqa.evaluation.metrics import AVAILABLE_METRICS

        unknown = [m for m in self.metrics if m not in AVAILABLE_METRICS]
        if unknown:
            raise ConfigError(
                f"{path}.metrics contains unknown metric(s): {unknown}. "
                f"Available: {sorted(AVAILABLE_METRICS)}"
            )


@dataclass
class Config:
    project_name: str = "FVQA"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        self.data.validate("data")
        self.retrieval.validate("retrieval")
        self.model.validate("model")
        self.trainer.validate("trainer")
        self.training.validate("training")
        self.inference.validate("inference")
        self.evaluation.validate("evaluation")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _coerce(value: Any, target: Any, path: str) -> Any:
    """Coerce a YAML scalar to the type declared on the dataclass field."""
    origin = typing.get_origin(target)

    # Optional[X] / X | None
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(target) if a is not type(None)]
        if value is None:
            return None
        return _coerce(value, args[0], path)

    if origin in (list, Sequence):
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list, got {type(value).__name__}")
        (item_type,) = typing.get_args(target) or (str,)
        return [_coerce(item, item_type, f"{path}[{i}]") for i, item in enumerate(value)]

    if target is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be a boolean, got {value!r}")
        return value

    if target is int:
        # A YAML `1e5` parses as float; accept it when it is integral.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be an integer, got {value!r}")
        if isinstance(value, float) and not value.is_integer():
            raise ConfigError(f"{path} must be an integer, got {value!r}")
        return int(value)

    if target is float:
        # YAML 1.1 only recognises scientific notation with a decimal point
        # and a signed exponent, so `2e-5` parses as the *string* "2e-5"
        # while `2.0e-5` parses as a float. Both spellings are natural to
        # write for a learning rate, so accept a numeric string here rather
        # than rejecting a config that looks perfectly correct.
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ConfigError(f"{path} must be a number, got {value!r}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be a number, got {value!r}")
        return float(value)

    if target is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string, got {value!r}")
        return value

    return value


@functools.lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    """Resolved type hints for a dataclass.

    `from __future__ import annotations` turns every annotation into a
    string, so `dataclasses.fields(...).type` cannot be inspected directly.
    """
    return typing.get_type_hints(cls)


def _build(cls: type, mapping: Any, path: str) -> Any:
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{path} must be a mapping, got {type(mapping).__name__}")

    hints = _hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(mapping) - set(fields))
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {path or 'config'}: {unknown}. "
            f"Known keys: {sorted(fields)}"
        )

    kwargs: dict[str, Any] = {}
    for name in fields:
        if name not in mapping:
            continue
        child_path = f"{path}.{name}" if path else name
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, mapping[name], child_path)
        else:
            kwargs[name] = _coerce(mapping[name], hint, child_path)
    return cls(**kwargs)


def default_config_path() -> Path:
    """Locate `config/config.yaml`, honouring $FVQA_CONFIG.

    Searched, in order: $FVQA_CONFIG, ./config/config.yaml, then the
    repository root inferred from this file's location — so the CLI works
    from a subdirectory and from an installed package alike.
    """
    env = os.environ.get("FVQA_CONFIG")
    if env:
        return Path(env)

    candidates = [
        Path.cwd() / "config" / "config.yaml",
        Path(__file__).resolve().parents[2] / "config" / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply `dotted.key=value` overrides onto a raw config mapping.

    Values are parsed as YAML, so `--set training.bf16=false` yields a
    boolean and `--set data.splits.val=0.1` yields a float.
    """
    for override in overrides:
        if "=" not in override:
            raise ConfigError(
                f"override {override!r} is not of the form key=value"
            )
        key, _, value = override.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"override {override!r} has an empty key")

        parts = key.split(".")
        cursor: dict[str, Any] = raw
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = yaml.safe_load(value)
    return raw


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Sequence[str] | None = None,
) -> Config:
    """Load, override and validate the configuration.

    Args:
        path: Path to the YAML file. Defaults to :func:`default_config_path`.
        overrides: `dotted.key=value` strings applied before validation.

    Raises:
        ConfigError: the file is missing, malformed, or holds an invalid value.
    """
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a mapping at the top level")

    if overrides:
        raw = apply_overrides(raw, overrides)

    config = _build(Config, raw, "")
    config.validate()
    return config
