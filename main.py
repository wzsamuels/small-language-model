import os

# Import the functions from your newly organized modules
# (This assumes you wrapped the logic in these files into callable functions)
from data_prep.download_datasets import download_soda_data
from data_prep.download_datasets import download_oasst_data
from data_prep.extract_gutenberg import process_gutenberg_plays
from data_prep.extract_soda import format_soda_data
from data_prep.extract_oasst import format_oasst_data
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
    
    # Setup
    setup_directories()
    
    
    print("\n--- Phase 1: Data Download ---")

    download_soda_data(output_file="data/raw/soda_raw.jsonl")
    download_oasst_data(output_file="data/raw/oasst_raw.jsonl")

    # Extraction Phase
    print("\n--- Phase 2: Data Extraction ---")
    # Hypothetical function calls based on your data_prep scripts
    #process_gutenberg_plays(input_dir="data/raw/gutenberg_plays", output_file="data/raw/gutenberg_formatted.jsonl")
    #download_and_format_reddit(output_file="data/raw/reddit_formatted.jsonl")
   
    format_soda_data(input_file="data/raw/soda_raw.jsonl", output_file="data/raw/soda_formatted.jsonl")
    format_oasst_data(input_file="data/raw/oasst_raw.jsonl", output_file="data/raw/oasst_formatted.jsonl")

    # 3. Blending Phase
    #print("\n--- Phase 2: Data Blending ---")
    #balance_and_blend_data(
    #    gutenberg_file="data/raw/gutenberg_formatted.jsonl",
    #    reddit_file="data/raw/reddit_formatted.jsonl",
    #    output_file="data/processed/master_training_data.jsonl",
    #    target_gutenberg_ratio=0.75
    #)
    
    # 4. Tokenization Phase
    print("\n--- Phase 3: Tokenizer Training ---")
    train_custom_tokenizer(
        dataset_path="data/raw/soda_formatted.jsonl",
        output_path="data/tokenizer.json",
        vocab_size=32000
    )
    
    print("\n=== Pipeline Complete! Ready for train.py ===")

    print("\n--- Phase 4: Model Training ---")
    train_model(input_training_file="data/raw/soda_formatted.jsonl", input_tokenizer_file="data/tokenizer.json", device="cuda", max_len=512)


if __name__ == "__main__":
    run_data_pipeline()
