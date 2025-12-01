"""
Vi-VQA Training on Modal
========================
Script để train Qwen3-VL trên Modal platform.

Usage:
    # 1. Install Modal CLI
    pip install modal

    # 2. Login to Modal
    modal token new

    # 3. Create HuggingFace secret
    modal secret create huggingface-secret HF_TOKEN=hf_your_token_here

    # 4. Run training pipeline
    modal run scripts/train_on_modal.py

    # 5. Run specific function
    modal run scripts/train_on_modal.py::prepare_dataset
    modal run scripts/train_on_modal.py::train_model
    modal run scripts/train_on_modal.py::evaluate_model

Training time (estimated):
    - A10G: ~20-25 giờ
    - A100: ~8-12 giờ
    - H100: ~4-6 giờ
"""

import modal
import os

# ============================================
# Modal App Configuration
# ============================================
app = modal.App("vi-vqa-training")

# Container image with all dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("git", "wget", "unzip")
    .pip_install(
        # Core ML
        "torch>=2.0.0",
        "torchvision",
        # Transformers & Training
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "peft>=0.13.0",
        "bitsandbytes>=0.44.0",
        "deepspeed>=0.15.0",
        # Data processing
        "datasets>=3.0.0",
        "pillow>=10.0.0",
        "pyyaml",
        "scipy",
        "tqdm",
        # Qwen-VL specific
        "qwen-vl-utils>=0.0.8",
        # HuggingFace
        "huggingface_hub>=0.26.0",
        # Evaluation
        "nltk",
        "rouge-score",
        # Additional
        "wandb",
        "tensorboard",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .run_commands(
        # Install flash-attention (optional, may fail on some GPUs)
        "pip install flash-attn --no-build-isolation || echo 'Flash attention not available'"
    )
)

# Persistent volumes for data and checkpoints
data_volume = modal.Volume.from_name("vi-vqa-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("vi-vqa-checkpoints", create_if_missing=True)

# Volume mount paths
DATA_DIR = "/data"
CHECKPOINT_DIR = "/checkpoints"
IMAGE_FOLDER = "/data/images"


# ============================================
# Step 1: Prepare Dataset
# ============================================
@app.function(
    image=image,
    volumes={DATA_DIR: data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,  # 1 hour
    memory=8192,   # 8GB RAM
)
def prepare_dataset():
    """
    Download and process Viet-ViTextVQA dataset.
    
    Output:
        - /data/train.json (~28K samples)
        - /data/val.json (~3K samples)
        - /data/images/ (~9.5K images)
    """
    from datasets import load_dataset
    from huggingface_hub import login
    from PIL import Image
    import json
    from tqdm import tqdm
    import random

    # Login to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✓ Logged in to HuggingFace")

    # Create directories
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    # Check if already processed
    train_path = f"{DATA_DIR}/train.json"
    val_path = f"{DATA_DIR}/val.json"
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        with open(train_path, 'r') as f:
            train_count = len(json.load(f))
        with open(val_path, 'r') as f:
            val_count = len(json.load(f))
        print(f"✓ Dataset already prepared: {train_count} train, {val_count} val")
        data_volume.commit()
        return {"train": train_count, "val": val_count, "status": "cached"}

    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("5CD-AI/Viet-ViTextVQA-gemini-VQA", split="train")
    print(f"✓ Loaded {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")

    # Process dataset
    processed_samples = []
    
    print("Processing dataset...")
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]
        image = item["image"]
        conversations = item.get("conversations", [])

        if not conversations:
            continue

        # Save image
        image_filename = f"image_{item['id']}.jpg"
        image_path = os.path.join(IMAGE_FOLDER, image_filename)

        if not os.path.exists(image_path):
            try:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, quality=95)
            except Exception as e:
                print(f"Warning: Failed to save image {idx}: {e}")
                continue

        # Process conversations
        current_question = None
        for turn in conversations:
            role = turn.get("role", turn.get("from"))
            content = turn.get("content", turn.get("value"))

            if role in ["user", "human"]:
                current_question = content
            elif role in ["assistant", "gpt"] and current_question:
                processed_samples.append({
                    "id": f"{item['id']}_{len(processed_samples)}",
                    "image": image_filename,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{current_question}"},
                        {"from": "gpt", "value": content}
                    ]
                })
                current_question = None

    print(f"✓ Processed {len(processed_samples)} QA pairs")

    # Split dataset: 90% train, 10% validation
    random.seed(42)
    shuffled_samples = processed_samples.copy()
    random.shuffle(shuffled_samples)

    split_idx = int(len(shuffled_samples) * 0.9)
    train_samples = shuffled_samples[:split_idx]
    val_samples = shuffled_samples[split_idx:]

    print(f"\n📊 Dataset Split:")
    print(f"  Train: {len(train_samples)} samples ({len(train_samples)/len(processed_samples)*100:.1f}%)")
    print(f"  Val:   {len(val_samples)} samples ({len(val_samples)/len(processed_samples)*100:.1f}%)")

    # Save datasets
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved train set to {train_path}")

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved validation set to {val_path}")

    # Count images
    num_images = len(os.listdir(IMAGE_FOLDER))
    print(f"✓ Saved {num_images} images to {IMAGE_FOLDER}")

    # Commit volume changes
    data_volume.commit()

    return {
        "train": len(train_samples),
        "val": len(val_samples),
        "images": num_images,
        "status": "created"
    }


# ============================================
# Debug: Test Environment
# ============================================
@app.function(
    image=image,
    gpu="A100",
    volumes={
        DATA_DIR: data_volume,
        CHECKPOINT_DIR: checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=32768,
)
def test_environment():
    """
    Test if the training environment is properly set up.
    This helps debug issues before running full training.
    """
    import subprocess
    import sys
    from huggingface_hub import login
    
    print("=" * 80)
    print("🔍 Testing Training Environment")
    print("=" * 80)
    
    # 1. Check GPU
    print("\n1. GPU Info:")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"])
    
    # 2. Check Python packages
    print("\n2. Key Packages:")
    import torch
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    import transformers
    print(f"   Transformers: {transformers.__version__}")
    
    try:
        import peft
        print(f"   PEFT: {peft.__version__}")
    except:
        print("   PEFT: NOT INSTALLED ⚠️")
    
    try:
        import bitsandbytes
        print(f"   BitsAndBytes: {bitsandbytes.__version__}")
    except:
        print("   BitsAndBytes: NOT INSTALLED ⚠️")
    
    # 3. Login to HuggingFace
    print("\n3. HuggingFace Login:")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("   ✓ Logged in successfully")
    else:
        print("   ⚠️ HF_TOKEN not found")
    
    # 4. Check data files
    print("\n4. Data Files:")
    train_path = f"{DATA_DIR}/train.json"
    val_path = f"{DATA_DIR}/val.json"
    print(f"   Train exists: {os.path.exists(train_path)}")
    print(f"   Val exists: {os.path.exists(val_path)}")
    print(f"   Images folder exists: {os.path.exists(IMAGE_FOLDER)}")
    if os.path.exists(IMAGE_FOLDER):
        print(f"   Number of images: {len(os.listdir(IMAGE_FOLDER))}")
    
    # 5. Clone and check training repo
    print("\n5. Training Repository:")
    repo_dir = "/root/Qwen-VL-Series-Finetune"
    if not os.path.exists(repo_dir):
        print("   Cloning repository...")
        subprocess.run([
            "git", "clone",
            "https://github.com/2U1/Qwen-VL-Series-Finetune.git",
            repo_dir
        ], check=True)
    
    print(f"   Repo exists: {os.path.exists(repo_dir)}")
    
    # List files in repo
    print(f"   Files in repo:")
    for f in os.listdir(repo_dir)[:10]:
        print(f"      - {f}")
    
    # Check if train.py exists
    train_script = os.path.join(repo_dir, "src", "train", "train_sft.py")
    print(f"   src/train/train_sft.py exists: {os.path.exists(train_script)}")
    
    # List scripts folder
    scripts_dir = os.path.join(repo_dir, "scripts")
    if os.path.exists(scripts_dir):
        print(f"   Scripts in scripts/:")
        for f in os.listdir(scripts_dir)[:10]:
            print(f"      - {f}")
    
    # List src/train folder
    train_dir = os.path.join(repo_dir, "src", "train")
    if os.path.exists(train_dir):
        print(f"   Files in src/train/:")
        for f in os.listdir(train_dir)[:10]:
            print(f"      - {f}")
    
    # 6. Try to import model (dry run)
    print("\n6. Model Loading Test:")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
        print(f"   ✓ Can access model config")
        print(f"   Model type: {config.model_type}")
    except Exception as e:
        print(f"   ⚠️ Error accessing model: {e}")
    
    # 7. Check deepspeed
    print("\n7. Checking deepspeed:")
    try:
        import deepspeed
        print(f"   ✓ Deepspeed version: {deepspeed.__version__}")
    except Exception as e:
        print(f"   ⚠️ Deepspeed error: {e}")
    
    print("\n" + "=" * 80)
    print("🔍 Environment Test Complete")
    print("=" * 80)
    
    return {"status": "ok"}


# ============================================
# Step 2: Training
# ============================================
@app.function(
    image=image,
    gpu="A100",  # Options: "T4", "A10G", "A100", "H100"
    volumes={
        DATA_DIR: data_volume,
        CHECKPOINT_DIR: checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,  # 24 hours
    memory=32768,   # 32GB RAM
)
def train_model(
    num_epochs: int = 2,
    batch_size: int = 1,
    grad_accum: int = 16,
    learning_rate: float = 2e-5,
    lora_rank: int = 128,
    resume_from_checkpoint: bool = True,
    debug: bool = False,
):
    """
    Fine-tune Qwen3-VL with LoRA on Vi-VQA dataset.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        grad_accum: Gradient accumulation steps
        learning_rate: Learning rate for LLM
        lora_rank: LoRA rank (higher = more capacity)
        resume_from_checkpoint: Resume from last checkpoint if exists
    """
    import subprocess
    from huggingface_hub import login

    # Login to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✓ Logged in to HuggingFace")

    # Check if data exists
    train_path = f"{DATA_DIR}/train.json"
    val_path = f"{DATA_DIR}/val.json"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run prepare_dataset first: modal run scripts/train_on_modal.py::prepare_dataset"
        )

    # Clone training repository
    repo_dir = "/root/Qwen-VL-Series-Finetune"
    if not os.path.exists(repo_dir):
        print("Cloning Qwen-VL-Series-Finetune repository...")
        subprocess.run([
            "git", "clone",
            "https://github.com/2U1/Qwen-VL-Series-Finetune.git",
            repo_dir
        ], check=True)
        print("✓ Cloned training repository")
        
        # Install repo dependencies
        print("Installing training repository dependencies...")
        subprocess.run(
            ["pip", "install", "-e", repo_dir],
            check=False,
            capture_output=True
        )
        
        # Install additional requirements if exists
        req_file = os.path.join(repo_dir, "requirements.txt")
        if os.path.exists(req_file):
            subprocess.run(
                ["pip", "install", "-r", req_file],
                check=False,
                capture_output=True
            )
        print("✓ Installed dependencies")
    else:
        print("✓ Training repository already exists")

    os.chdir(repo_dir)

    # Training configuration
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    OUTPUT_DIR = f"{CHECKPOINT_DIR}/qwen3vl-vivqa"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Image resolution (Qwen3-VL uses 32x32 patches)
    IMAGE_MIN_PIXELS = 256 * 32 * 32   # 262144
    IMAGE_MAX_PIXELS = 1280 * 32 * 32  # 1310720

    # Build training command using deepspeed (the correct way for this repo)
    # The repo uses src/train/train_sft.py for SFT training
    train_cmd = [
        "deepspeed",
        "--num_gpus=1",  # Single GPU on Modal
        "src/train/train_sft.py",
        # LoRA config
        "--lora_enable", "True",
        "--use_dora", "False",
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_rank * 2),
        "--lora_dropout", "0.05",
        "--num_lora_modules", "-1",  # All modules
        "--lora_namespan_exclude", "['lm_head', 'embed_tokens']",
        # Deepspeed config
        "--deepspeed", "scripts/zero2.json",
        # Model
        "--model_id", MODEL_ID,
        # Data
        "--data_path", train_path,
        "--eval_data_path", val_path,
        "--image_folder", IMAGE_FOLDER,
        "--output_dir", OUTPUT_DIR,
        "--remove_unused_columns", "False",
        # Freezing
        "--freeze_vision_tower", "True",
        "--freeze_llm", "False",  # Train LLM with LoRA
        "--freeze_merger", "False",
        # Precision
        "--bf16", "True",
        "--fp16", "False",
        "--tf32", "True",
        # Training params
        "--num_train_epochs", str(num_epochs),
        "--per_device_train_batch_size", str(batch_size),
        "--per_device_eval_batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        # Image config (Qwen3-VL uses 32x32 patches)
        "--image_min_pixels", str(IMAGE_MIN_PIXELS),
        "--image_max_pixels", str(IMAGE_MAX_PIXELS),
        # Learning rates
        "--learning_rate", str(learning_rate),
        "--vision_lr", str(learning_rate / 10),
        "--merger_lr", str(learning_rate),
        # Optimization
        "--weight_decay", "0.1",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--gradient_checkpointing", "True",
        "--max_grad_norm", "1.0",
        "--dataloader_num_workers", "4",
        # Disable flash attention (not installed)
        "--disable_flash_attn2", "True",
        # Evaluation & Saving
        "--eval_strategy", "steps",
        "--eval_steps", "500",
        "--save_strategy", "steps",
        "--save_steps", "500",
        "--save_total_limit", "3",
        "--load_best_model_at_end", "True",
        "--metric_for_best_model", "eval_loss",
        "--greater_is_better", "False",
        # Logging
        "--logging_steps", "50",
        "--report_to", "tensorboard",
        # Other
        "--lazy_preprocess", "True",
    ]

    # Add resume flag if checkpoint exists
    if resume_from_checkpoint and os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            train_cmd.extend(["--resume_from_checkpoint", "true"])
            print(f"📂 Found checkpoints: {checkpoints}")
            print("   Will resume from latest checkpoint")

    print("\n" + "=" * 80)
    print("🚀 Starting Training")
    print("=" * 80)
    print(f"Model: {MODEL_ID}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size} × {grad_accum} = {batch_size * grad_accum}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    # Run training with full output capture
    print("Running training command...")
    print(f"Command: {' '.join(train_cmd[:10])}...")  # Print first part of command
    
    result = subprocess.run(
        train_cmd, 
        check=False,
        text=True,
        # Capture and stream output
    )
    
    if result.returncode != 0:
        print(f"\n⚠️ Training exited with code {result.returncode}")
        
        # Try to find error logs
        log_dir = os.path.join(OUTPUT_DIR, "runs")
        if os.path.exists(log_dir):
            print(f"Check logs at: {log_dir}")
    else:
        print("\n✓ Training completed successfully!")

    # Commit checkpoint volume
    checkpoint_volume.commit()

    # List saved checkpoints
    if os.path.exists(OUTPUT_DIR):
        checkpoints = sorted([d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")])
        print(f"\n📦 Saved checkpoints: {checkpoints}")
        return {"status": "completed", "checkpoints": checkpoints}
    
    return {"status": "completed", "checkpoints": []}


# ============================================
# Step 3: Evaluation
# ============================================
@app.function(
    image=image,
    gpu="A100",
    volumes={
        DATA_DIR: data_volume,
        CHECKPOINT_DIR: checkpoint_volume,
    },
    timeout=7200,  # 2 hours
    memory=32768,
)
def evaluate_model(
    checkpoint_name: str = "checkpoint-best",
    num_samples: int = 100,
    temperature: float = 0.7,
):
    """
    Evaluate trained model on validation set.
    
    Args:
        checkpoint_name: Name of checkpoint folder (e.g., "checkpoint-1500")
        num_samples: Number of samples to evaluate (use -1 for all)
        temperature: Generation temperature
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from difflib import SequenceMatcher
    import json
    from tqdm import tqdm

    # Find checkpoint path
    checkpoint_base = f"{CHECKPOINT_DIR}/qwen3vl-vivqa"
    
    if checkpoint_name == "checkpoint-best":
        # Find the latest/best checkpoint
        if os.path.exists(checkpoint_base):
            checkpoints = sorted([
                d for d in os.listdir(checkpoint_base) 
                if d.startswith("checkpoint-")
            ])
            if checkpoints:
                checkpoint_name = checkpoints[-1]
                print(f"📂 Using latest checkpoint: {checkpoint_name}")
            else:
                raise FileNotFoundError("No checkpoints found!")
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_base}")

    checkpoint_path = os.path.join(checkpoint_base, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    
    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    print("✓ Model loaded!")

    # Load validation data
    val_path = f"{DATA_DIR}/val.json"
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    if num_samples > 0:
        val_data = val_data[:num_samples]
    
    print(f"\n📊 Evaluating on {len(val_data)} samples...")

    def ask_question(image_path, question):
        """Generate answer for a question about an image."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }]

        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.9,
                do_sample=True if temperature > 0 else False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        answer = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Clean up
        del inputs, generated_ids
        torch.cuda.empty_cache()

        return answer

    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # Evaluate
    exact_matches = 0
    similarities = []
    results = []

    for i in tqdm(range(len(val_data))):
        sample = val_data[i]
        image_path = os.path.join(IMAGE_FOLDER, sample["image"])
        question = sample["conversations"][0]["value"].replace("<image>\n", "")
        ground_truth = sample["conversations"][1]["value"]

        try:
            prediction = ask_question(image_path, question)

            # Calculate metrics
            is_exact = prediction.strip() == ground_truth.strip()
            sim = similarity(prediction, ground_truth)

            if is_exact:
                exact_matches += 1
            similarities.append(sim)

            results.append({
                "id": sample["id"],
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "exact_match": is_exact,
                "similarity": sim
            })

            # Print first 3 examples
            if i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Q: {question}")
                print(f"GT: {ground_truth}")
                print(f"Pred: {prediction}")
                print(f"Similarity: {sim*100:.1f}%")

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    # Calculate final metrics
    num_evaluated = len(similarities)
    exact_match_rate = exact_matches / num_evaluated * 100 if num_evaluated > 0 else 0
    avg_similarity = sum(similarities) / num_evaluated * 100 if num_evaluated > 0 else 0

    print(f"\n{'=' * 80}")
    print("📊 Evaluation Results")
    print(f"{'=' * 80}")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Samples evaluated: {num_evaluated}")
    print(f"Exact Match: {exact_matches}/{num_evaluated} ({exact_match_rate:.2f}%)")
    print(f"Avg Similarity: {avg_similarity:.2f}%")
    print(f"{'=' * 80}")

    # Save results
    results_path = f"{CHECKPOINT_DIR}/eval_results_{checkpoint_name}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint": checkpoint_name,
            "num_samples": num_evaluated,
            "exact_match": exact_match_rate,
            "avg_similarity": avg_similarity,
            "results": results[:20]  # Save first 20 detailed results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    checkpoint_volume.commit()

    return {
        "checkpoint": checkpoint_name,
        "num_samples": num_evaluated,
        "exact_match": exact_match_rate,
        "avg_similarity": avg_similarity,
    }


# ============================================
# Step 4: Interactive Demo
# ============================================
@app.function(
    image=image,
    gpu="A100",
    volumes={
        DATA_DIR: data_volume,
        CHECKPOINT_DIR: checkpoint_volume,
    },
    timeout=3600,
)
def demo(
    image_path: str,
    question: str,
    checkpoint_name: str = "checkpoint-best",
):
    """
    Run inference on a single image.
    
    Args:
        image_path: Path to image file (local or URL)
        question: Question to ask about the image
        checkpoint_name: Name of checkpoint to use
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Find checkpoint
    checkpoint_base = f"{CHECKPOINT_DIR}/qwen3vl-vivqa"
    
    if checkpoint_name == "checkpoint-best":
        checkpoints = sorted([
            d for d in os.listdir(checkpoint_base) 
            if d.startswith("checkpoint-")
        ])
        checkpoint_name = checkpoints[-1] if checkpoints else None
        if not checkpoint_name:
            raise FileNotFoundError("No checkpoints found!")

    checkpoint_path = os.path.join(checkpoint_base, checkpoint_name)

    print(f"Loading model from {checkpoint_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(checkpoint_path)

    # Prepare message
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question}
        ]
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    print(f"\n{'=' * 60}")
    print(f"📸 Image: {image_path}")
    print(f"❓ Question: {question}")
    print(f"💬 Answer: {answer}")
    print(f"{'=' * 60}")

    return {"question": question, "answer": answer}


# ============================================
# Step 5: Download Checkpoint
# ============================================
@app.function(
    image=image,
    volumes={CHECKPOINT_DIR: checkpoint_volume},
    timeout=1800,
)
def download_checkpoint(checkpoint_name: str = "checkpoint-best"):
    """
    Download a checkpoint as a zip file.
    
    Args:
        checkpoint_name: Name of checkpoint to download
    """
    import subprocess
    import shutil

    checkpoint_base = f"{CHECKPOINT_DIR}/qwen3vl-vivqa"
    
    if checkpoint_name == "checkpoint-best":
        checkpoints = sorted([
            d for d in os.listdir(checkpoint_base) 
            if d.startswith("checkpoint-")
        ])
        checkpoint_name = checkpoints[-1] if checkpoints else None

    checkpoint_path = os.path.join(checkpoint_base, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create zip file
    zip_path = f"/tmp/{checkpoint_name}.zip"
    print(f"Creating zip archive: {zip_path}")
    
    shutil.make_archive(
        zip_path.replace(".zip", ""),
        "zip",
        checkpoint_base,
        checkpoint_name
    )

    # Read and return zip content
    with open(zip_path, "rb") as f:
        content = f.read()

    print(f"✓ Checkpoint ready for download: {len(content) / 1e6:.2f} MB")
    
    return content


# ============================================
# Main Entry Point
# ============================================
@app.local_entrypoint()
def main(
    step: str = "all",
    num_epochs: int = 2,
    batch_size: int = 1,
    grad_accum: int = 16,
    num_eval: int = 100,
):
    """
    Run the full training pipeline.
    
    Args:
        step: Which step to run ("prepare", "train", "evaluate", "all")
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        grad_accum: Gradient accumulation steps
        num_eval: Number of samples to evaluate
    
    Examples:
        modal run scripts/train_on_modal.py --step prepare
        modal run scripts/train_on_modal.py --step train --num-epochs 2
        modal run scripts/train_on_modal.py --step evaluate --num-eval 100
        modal run scripts/train_on_modal.py --step all
    """
    print("=" * 80)
    print("🚀 Vi-VQA Training Pipeline on Modal")
    print("=" * 80)

    if step in ["prepare", "all"]:
        print("\n📦 Step 1: Preparing dataset...")
        result = prepare_dataset.remote()
        print(f"   Result: {result}")

    if step in ["train", "all"]:
        print("\n🎯 Step 2: Training model...")
        result = train_model.remote(
            num_epochs=num_epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
        )
        print(f"   Result: {result}")

    if step in ["evaluate", "all"]:
        print("\n📊 Step 3: Evaluating model...")
        result = evaluate_model.remote(num_samples=num_eval)
        print(f"   Result: {result}")

    print("\n" + "=" * 80)
    print("🎉 Pipeline complete!")
    print("=" * 80)


# ============================================
# Utility Functions
# ============================================
@app.function(
    volumes={
        DATA_DIR: data_volume,
        CHECKPOINT_DIR: checkpoint_volume,
    },
)
def check_status():
    """Check the status of data and checkpoints."""
    import json

    status = {"data": {}, "checkpoints": {}}

    # Check data
    train_path = f"{DATA_DIR}/train.json"
    val_path = f"{DATA_DIR}/val.json"
    
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            status["data"]["train"] = len(json.load(f))
    
    if os.path.exists(val_path):
        with open(val_path, 'r') as f:
            status["data"]["val"] = len(json.load(f))

    if os.path.exists(IMAGE_FOLDER):
        status["data"]["images"] = len(os.listdir(IMAGE_FOLDER))

    # Check checkpoints
    checkpoint_base = f"{CHECKPOINT_DIR}/qwen3vl-vivqa"
    if os.path.exists(checkpoint_base):
        checkpoints = sorted([
            d for d in os.listdir(checkpoint_base) 
            if d.startswith("checkpoint-")
        ])
        status["checkpoints"]["available"] = checkpoints

    print("\n📊 Status:")
    print(f"   Data: {status['data']}")
    print(f"   Checkpoints: {status['checkpoints']}")

    return status
