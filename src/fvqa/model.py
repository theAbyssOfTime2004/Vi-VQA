"""Loading Qwen3-VL and answering questions with it.

Two things the previous inference code got wrong, both of which surface
only once a GPU is attached and a model has been downloaded:

1. It imported `Qwen2VLForConditionalGeneration` for a Qwen3-VL
   checkpoint. Different architecture class, different config — the load
   fails, and the failure names a Qwen2 class for a Qwen3 model, which
   sends you looking in the wrong place.
2. It hard-coded `attn_implementation="flash_attention_2"`. When
   flash-attn is not installed — the Modal image explicitly gives up on
   it — every load raises, even though SDPA attention would have worked.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from fvqa.config import Config
from fvqa.data.grounding import apply_grounding

__all__ = ["VQAModel", "load_model", "resolve_model_class"]

logger = logging.getLogger(__name__)


def resolve_model_class() -> Any:
    """Return the class that loads a Qwen3-VL checkpoint.

    Raises:
        ImportError: transformers is missing or too old to know about
            Qwen3-VL at all.
    """
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "transformers is required to load the model. "
            "Install it with: pip install 'fvqa[infer]'"
        ) from error

    model_class = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_class is not None:
        return model_class

    # Newer transformers route Qwen3-VL through the generic auto class.
    auto_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if auto_class is not None:
        logger.warning(
            "Qwen3VLForConditionalGeneration not found in transformers %s; "
            "falling back to AutoModelForImageTextToText",
            transformers.__version__,
        )
        return auto_class

    raise ImportError(
        f"transformers {transformers.__version__} is too old for Qwen3-VL. "
        "Upgrade with: pip install -U 'transformers>=4.57.0'"
    )


def _torch_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"unknown torch dtype: {name!r}")
    return dtype


def load_model(model_path: str, config: Config, device_map: str = "auto") -> tuple[Any, Any]:
    """Load a checkpoint and its processor.

    Falls back from flash attention to SDPA when flash-attn is unavailable,
    rather than failing the run over an optional speedup.

    Returns:
        (model, processor)
    """
    # resolve_model_class() first: it raises the actionable message when
    # transformers is missing or too old, rather than a bare ImportError.
    model_class = resolve_model_class()
    dtype = _torch_dtype(config.model.torch_dtype)

    attn_implementations = ["flash_attention_2", "sdpa"] if config.model.use_flash_attn else ["sdpa"]

    last_error: Exception | None = None
    for attn in attn_implementations:
        try:
            logger.info("loading %s (attn=%s)", model_path, attn)
            model = model_class.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=device_map,
                attn_implementation=attn,
            )
            break
        except (ImportError, ValueError) as error:
            logger.warning("could not load with attn_implementation=%s: %s", attn, error)
            last_error = error
    else:  # every implementation failed
        raise RuntimeError(f"failed to load model from {model_path}") from last_error

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


@dataclass
class VQAModel:
    """A loaded model plus the prompt conventions used to train it.

    Grounding is applied through the same `apply_grounding` the data
    pipeline uses. That matters: a model fine-tuned on grounded prompts
    and queried with bare ones sees a prompt format it never saw in
    training, and answers noticeably worse for reasons that look like a
    modelling problem rather than a plumbing one.
    """

    model: Any
    processor: Any
    config: Config

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Config,
        device_map: str = "auto",
    ) -> "VQAModel":
        model, processor = load_model(model_path, config, device_map=device_map)
        return cls(model=model, processor=processor, config=config)

    def build_messages(
        self,
        image_path: str,
        question: str,
        description: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the chat messages for one question."""
        prompt = apply_grounding(question, description, self.config.data.grounding)

        messages: list[dict[str, Any]] = []
        style = self.config.inference.system_prompt
        if style:
            messages.append({"role": "system", "content": [{"type": "text", "text": style}]})
        if prompt.system:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": prompt.system}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt.question},
                ],
            }
        )
        return messages

    def answer(
        self,
        image_path: str,
        question: str,
        description: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """Answer one question about one image, applying grounding."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"image not found: {image_path}")

        messages = self.build_messages(image_path, question, description)
        return self.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """Run generation on already-built messages.

        Evaluation replays the exact prompt stored in the split file,
        grounding included, so it must not go through `build_messages` —
        that would apply grounding a second time and score the model on a
        prompt it was never trained on.

        `temperature=0` selects greedy decoding, which is what evaluation
        should use: sampled answers make a metric move between runs on an
        unchanged model.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        inference = self.config.inference
        max_new_tokens = max_new_tokens if max_new_tokens is not None else inference.max_new_tokens
        temperature = temperature if temperature is not None else inference.temperature
        top_p = top_p if top_p is not None else inference.top_p

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            generate_kwargs.update(
                do_sample=True, temperature=temperature, top_p=top_p
            )
        else:
            generate_kwargs["do_sample"] = False

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)

        trimmed = [
            output[len(prompt_ids) :]
            for prompt_ids, output in zip(inputs.input_ids, generated)
        ]
        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Long eval loops otherwise accumulate activations until the GPU
        # runs out of memory partway through.
        del inputs, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return answer.strip()
