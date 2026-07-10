import os
import json
from datasets import load_dataset
from tqdm import tqdm

def download_oasst_data(output_file="data/raw/oasst_raw.jsonl"):
    """Downloads the OpenAssistant (oasst2) dataset and saves it locally."""

    # Ensure the target directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping oasst download.")    
        return

    print("Downloading OpenAssistant/oasst2 from Hugging Face...")
    
    # Load the training split of the dataset
    # oasst2 only has roughly 128,000 messages, so we will download the entire thing
    dataset = load_dataset("OpenAssistant/oasst2", split="train")
    total_rows = len(dataset)

    print(f"Saving {total_rows:,} raw messages to {output_file}...")
    
    # Write the raw dictionary rows directly to a JSONL file with progress tracking
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in tqdm(dataset, desc="Writing data to file"):
            f.write(json.dumps(row) + '\n')
            
    print("Download complete!")

def download_soda_data(output_file="data/raw/soda_raw.jsonl", sample_size=None):
    """Downloads a sample of the SODA dataset and saves it to a local JSONL file."""
    
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping SODA download.")    
        return

    print(f"Downloading {'{sample_size:,}' if sample_size else 'all'} rows from allenai/soda...")
    
    # Load the training split of the dataset from Hugging Face
    dataset = load_dataset("allenai/soda", split="train")

    # Shuffle the dataset and take a slice of our desired size
    sampled_dataset = dataset.shuffle(seed=42)
    if sample_size:
        sampled_dataset = sampled_dataset.select(range(sample_size))

    print(f"Saving raw SODA data to {output_file}...")
    
    # Write the raw dictionary rows directly to a JSONL file with progress tracking
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in tqdm(sampled_dataset, desc="Writing data to file"):
            f.write(json.dumps(row) + '\n')
            
    print("Download complete!")

if __name__ == "__main__":
    download_oasst_data()
    download_soda_data()
