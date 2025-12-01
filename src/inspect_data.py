import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_config, setup_logging
from src.dataset import ViVQADataset
import logging

def main():
    setup_logging()
    config = load_config("config/config.yaml")
    
    print("Initializing Dataset...")
    try:
        # Load a small split or just the train split
        dataset = ViVQADataset(config, split="train")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    print(f"Dataset loaded. Total samples: {len(dataset)}")
    print(f"Dataset features: {dataset.dataset.features}")
    
    # Create visualizations directory
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"Vocab size: {len(dataset.vocab)}")
    
    # Inspect first 5 samples
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image Shape: {sample['image'].shape}")
        print(f"  Input IDs Shape: {sample['input_ids'].shape}")
        print(f"  Attention Mask Shape: {sample['attention_mask'].shape}")
        print(f"  Label: {sample['label']} (Word: {dataset.vocab.get_word(sample['label'].item())})")
        
        # Decode question
        question = dataset.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"  Decoded Question: {question}")
        
        # Save image (need to un-normalize to view properly, but for now just saving tensor as is might not work well for viewing, 
        # let's just rely on shape verification or inverse transform if needed. 
        # Actually, let's skip saving image for now as it's a tensor)
            
if __name__ == "__main__":
    main()
