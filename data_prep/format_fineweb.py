import os
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def format_fineweb_data(output_file="data/raw/fineweb_formatted.jsonl", num_articles=3500000):
    """
    Downloads articles from Hugging Face's high-quality FineWeb dataset 
    and chunks them for Phase 1 pre-training.
    """

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping FineWeb dataset download and formatting.")
        return

    print(f"Downloading {num_articles:,} FineWeb articles from Hugging Face...")
    
    # We specifically target the 'sample-10BT' subset so we aren't querying the massive 500GB dump!
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    
    # Safely limit the stream to our target number
    dataset = dataset.take(num_articles)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    chunk_size = 150 # Words per chunk (for the 256/512 context window)
    total_chunks = 0

    print(f"Chunking articles and saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for article in tqdm(dataset, total=num_articles, desc="Parsing FineWeb"):
            
            raw_text = article.get("text", "")
            if not raw_text:
                continue
                
            # Clean up escaped quotes from the web scraper
            clean_text = re.sub(r"\\+(['\"])", r"\1", raw_text)
            
            # Destroy newlines (\n) and normalize all spaces
            clean_text = " ".join(clean_text.split())
            words = clean_text.split()
            
            # Group into chunks of ~150 words
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                
                # Only save substantial chunks
                if len(chunk) > 50: 
                    out_file.write(json.dumps({"text": chunk}, ensure_ascii=False) + '\n')
                    total_chunks += 1

    print(f"\nExtraction Complete!")
    print(f"Successfully created {total_chunks:,} pre-training chunks from FineWeb.")

if __name__ == "__main__":
    format_fineweb_data()