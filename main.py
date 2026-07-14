import os

# Import the functions from your newly organized modules
# (This assumes you wrapped the logic in these files into callable functions)
from data_prep.download_datasets import download_soda_data
from data_prep.download_datasets import download_oasst_data
from data_prep.download_gutenberg import download_gutenberg_plays

from data_prep.format_gutenberg import format_gutenberg_plays
from data_prep.format_soda import format_soda_data
from data_prep.format_oasst import format_oasst_data
from data_prep.blend_datasets import blend_data
from data_prep.train_tokenizer import train_custom_tokenizer
from train import train_model

def setup_directories():
    """Ensure our data folders exist before scripts try to write to them."""
    directories = [
        "data/raw/gutenberg_plays",
        "data/raw/",
        "data/processed",
        "logs"
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)
        print(f"Verified directory: {d}")

def run_data_pipeline():
    print("=== Starting AI Pipeline ===")
    
    setup_directories()
    
    print("\n--- Phase 1: Data Download ---")

    download_gutenberg_plays(input_file="data/play_ids.txt", output_dir="data/raw/gutenberg_plays")
    download_soda_data(output_file="data/raw/soda_raw.jsonl")
    download_oasst_data(output_file="data/raw/oasst_raw.jsonl")

    print("\n--- Phase 2: Data Formatting ---")

    format_gutenberg_plays(input_dir="data/raw/gutenberg_plays", output_file="data/processed/gutenberg_formatted.jsonl")
    format_soda_data(input_file="data/raw/soda_raw.jsonl", output_file="data/processed/soda_formatted.jsonl")
    format_oasst_data(input_file="data/raw/oasst_raw.jsonl", output_file="data/processed/oasst_formatted.jsonl")
    
    print("\n--- Phase 3: Data Blending ---")
    blend_data(input_files = [
        "data/processed/gutenberg_formatted.jsonl",
        "data/processed/soda_formatted.jsonl",
        "data/processed/oasst_formatted.jsonl"
    ], output_path="data/processed/master_training_data.jsonl")

    print("\n--- Phase 4: Tokenizer Training ---")
    train_custom_tokenizer(
        dataset_path="data/processed/master_training_data.jsonl",
        output_path="data/tokenizer.json",
        vocab_size=32000
    )

    print("\n--- Phase 5: Model Training ---")
    train_model(input_training_file="data/processed/master_training_data.jsonl", input_tokenizer_file="data/tokenizer.json", device="cuda", max_len=512)


if __name__ == "__main__":
    run_data_pipeline()
