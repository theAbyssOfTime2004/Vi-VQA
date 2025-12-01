"""
Inference script for Qwen3-VL on Vi-VQA dataset
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_config


def load_model(model_path: str, device: str = "cuda"):
    """
    Load Qwen3-VL model and processor.

    Args:
        model_path: Path to model checkpoint or HuggingFace model ID
        device: Device to load model on

    Returns:
        model, processor
    """
    print(f"Loading model from {model_path}...")

    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"  # Use flash attention if available
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"✓ Model loaded on {device}")
    return model, processor


def inference(
    model,
    processor,
    image_path: str,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """
    Run inference on a single image-question pair.

    Args:
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        image_path: Path to image
        question: Question text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Generated answer text
    """
    # Prepare conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]

    # Prepare inputs
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
    )

    inputs = inputs.to(model.device)

    # Generate answer
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

    # Decode answer
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return answer


def interactive_demo(model_path: str):
    """
    Interactive demo for testing the model.

    Args:
        model_path: Path to model checkpoint
    """
    model, processor = load_model(model_path)

    print("\n" + "="*80)
    print("Qwen3-VL Vi-VQA Interactive Demo")
    print("="*80)
    print("Enter image path and question to get answer")
    print("Type 'quit' to exit")
    print("="*80 + "\n")

    while True:
        # Get image path
        image_path = input("Image path: ").strip()
        if image_path.lower() == 'quit':
            break

        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue

        # Get question
        question = input("Question: ").strip()
        if question.lower() == 'quit':
            break

        # Run inference
        print("\n🤔 Thinking...")
        try:
            answer = inference(model, processor, image_path, question)
            print(f"\n💡 Answer: {answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")

        print("-" * 80 + "\n")


def batch_evaluate(model_path: str, test_data_path: str, output_path: str = None):
    """
    Evaluate model on test dataset.

    Args:
        model_path: Path to model checkpoint
        test_data_path: Path to test JSON file
        output_path: Path to save predictions (optional)
    """
    import json

    model, processor = load_model(model_path)

    # Load test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"Evaluating on {len(test_data)} samples...")

    predictions = []

    for i, sample in enumerate(test_data):
        image_path = os.path.join("./data/images", sample['image'])

        # Extract question from conversations
        question = ""
        for conv in sample['conversations']:
            if conv['from'] == 'human':
                question = conv['value'].replace('<image>\n', '').strip()
                break

        # Get ground truth answer
        gt_answer = ""
        for conv in sample['conversations']:
            if conv['from'] == 'gpt':
                gt_answer = conv['value']
                break

        # Generate prediction
        try:
            pred_answer = inference(model, processor, image_path, question)

            predictions.append({
                'id': sample['id'],
                'question': question,
                'ground_truth': gt_answer,
                'prediction': pred_answer
            })

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(test_data)} samples")

        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")
            continue

    # Save predictions
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"✓ Predictions saved to {output_path}")

    # Calculate metrics (simple accuracy for now)
    correct = sum(1 for p in predictions if p['prediction'].strip() == p['ground_truth'].strip())
    accuracy = correct / len(predictions) * 100 if predictions else 0

    print(f"\n{'='*80}")
    print(f"Evaluation Results")
    print(f"{'='*80}")
    print(f"Total samples: {len(test_data)}")
    print(f"Processed: {len(predictions)}")
    print(f"Exact match accuracy: {accuracy:.2f}%")
    print(f"{'='*80}\n")

    return predictions


def main():
    """Main function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-VL Vi-VQA Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, choices=['interactive', 'eval'], default='interactive',
                        help="Mode: interactive or eval")
    parser.add_argument("--test_data", type=str, help="Path to test data JSON (for eval mode)")
    parser.add_argument("--output", type=str, help="Path to save predictions (for eval mode)")

    args = parser.parse_args()

    if args.mode == 'interactive':
        interactive_demo(args.model_path)
    elif args.mode == 'eval':
        if not args.test_data:
            print("❌ --test_data is required for eval mode")
            return
        batch_evaluate(args.model_path, args.test_data, args.output)


if __name__ == "__main__":
    main()
