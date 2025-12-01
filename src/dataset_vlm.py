"""
Dataset for Qwen3-VL Vision Language Model
Converts Vi-VQA data to Qwen3-VL conversation format
"""

import json
import os
from typing import Dict, Any, List
from datasets import load_dataset
from PIL import Image
import logging

class ViVQAVLMDataset:
    """
    Dataset class for Qwen3-VL format.

    Qwen3-VL expects conversations in format:
    [
        {"role": "user", "content": [{"type": "image", "image": <PIL.Image>}, {"type": "text", "text": "question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]}
    ]
    """

    def __init__(self, config: Dict[str, Any], split: str = "train"):
        """
        Initialize Vi-VQA Dataset for VLM.

        Args:
            config: Configuration dictionary
            split: Dataset split ('train', 'validation', 'test')
        """
        self.config = config
        self.split = split
        self.dataset_name = config['data']['dataset_name']
        self.data_dir = config['data']['data_dir']
        self.image_folder = config['data'].get('image_folder', os.path.join(self.data_dir, 'images'))

        # Create image folder if not exists
        os.makedirs(self.image_folder, exist_ok=True)

        logging.info(f"Loading dataset {self.dataset_name}, split: {split}")
        self.dataset = load_dataset(self.dataset_name, split=split)
        logging.info(f"Dataset columns: {self.dataset.column_names}")

        self.samples = self._process_dataset()
        logging.info(f"Processed {len(self.samples)} QA pairs for {split} split")

    def _process_dataset(self) -> List[Dict[str, Any]]:
        """
        Process dataset into Qwen3-VL format.

        Returns:
            List of processed samples with conversations
        """
        processed_samples = []

        for idx, item in enumerate(self.dataset):
            image = item['image']
            conversations = item.get('conversations', [])

            if not conversations:
                logging.warning(f"No conversations found for item {idx}")
                continue

            # Save image to disk (Qwen3-VL can also accept PIL images directly)
            image_filename = f"image_{item['id']}.jpg"
            image_path = os.path.join(self.image_folder, image_filename)

            if not os.path.exists(image_path):
                try:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(image_path)
                except Exception as e:
                    logging.warning(f"Failed to save image {idx}: {e}")
                    continue

            # Process conversations into QA pairs
            # Format: User -> Assistant -> User -> Assistant
            current_question = None

            for turn in conversations:
                role = turn.get('role', turn.get('from'))
                content = turn.get('content', turn.get('value'))

                if role in ['user', 'human']:
                    current_question = content

                elif role in ['assistant', 'gpt'] and current_question is not None:
                    # Create a conversation sample
                    # Qwen3-VL format with image in first user message
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": current_question}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": content}
                            ]
                        }
                    ]

                    processed_samples.append({
                        'id': f"{item['id']}_{len(processed_samples)}",
                        'image': image_path,
                        'conversations': conversation,
                        'original_index': idx
                    })

                    current_question = None

        return processed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return sample in Qwen3-VL format"""
        return self.samples[idx]

    def to_json(self, output_path: str):
        """
        Save dataset in JSON format for Qwen-VL-Series-Finetune script.

        Format expected:
        [
            {
                "id": "unique_id",
                "image": "image_filename.jpg",
                "conversations": [
                    {"from": "human", "value": "<image>\nQuestion?"},
                    {"from": "gpt", "value": "Answer."}
                ]
            }
        ]
        """
        json_data = []

        for sample in self.samples:
            # Convert to Qwen-VL-Series-Finetune format
            conversations_formatted = []

            # Extract question from user message
            user_msg = sample['conversations'][0]
            question_text = ""
            for content_item in user_msg['content']:
                if content_item['type'] == 'text':
                    question_text = content_item['text']
                    break

            # Extract answer from assistant message
            assistant_msg = sample['conversations'][1]
            answer_text = assistant_msg['content'][0]['text']

            conversations_formatted = [
                {"from": "human", "value": f"<image>\n{question_text}"},
                {"from": "gpt", "value": answer_text}
            ]

            json_data.append({
                "id": sample['id'],
                "image": os.path.basename(sample['image']),
                "conversations": conversations_formatted
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Saved {len(json_data)} samples to {output_path}")


def prepare_vlm_dataset(config: Dict[str, Any], splits: List[str] = ['train']):
    """
    Prepare VLM dataset and save to JSON format.

    Args:
        config: Configuration dictionary
        splits: List of splits to process
    """
    from src.utils import setup_logging
    setup_logging()

    for split in splits:
        logging.info(f"Processing {split} split...")
        dataset = ViVQAVLMDataset(config, split=split)

        # Save to JSON
        output_path = os.path.join(config['data']['data_dir'], f'{split}.json')
        dataset.to_json(output_path)

        logging.info(f"✓ {split} split prepared: {len(dataset)} samples")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from src.utils import load_config

    config = load_config('/home/maidang/projects/Vi-VQA/config/config.yaml')
    prepare_vlm_dataset(config, splits=['train'])
