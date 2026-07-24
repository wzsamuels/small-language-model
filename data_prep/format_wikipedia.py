import os
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def format_wikipedia_data(output_file="data/raw/wikipedia_formatted.jsonl", num_articles=5):
    """
    Downloads clean Wikipedia articles from Hugging Face and chunks them 
    for Phase 1 pre-training.
    """
    print(f"Downloading {num_articles:,} Wikipedia articles from Hugging Face...")
    
    # We use '20220301.en' which is a standard, clean English dump
    # We use streaming=True so we don't have to download the entire 20GB database to your drive!
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, token=os.getenv("HF_TOKEN"))
    dataset = dataset.take(num_articles)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    chunk_size = 150 # Words per chunk (leaves room in the 256 context window)
    total_chunks = 0

    print(f"Chunking articles and saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        # We manually iterate through the stream until we hit our desired number of articles
        for article in tqdm(dataset, total=num_articles, desc="Parsing Articles"):
                
            # The HF dataset stores the clean text in the "text" key
            raw_text = article.get("text", "")
            if not raw_text:
                continue
                
            # Clean up excessive newlines commonly found in wiki parsing
            clean_text = raw_text.replace('\\"','"')
            clean_text = " ".join(clean_text.split())
            words = clean_text.split()
            
            # Group into chunks of ~150 words
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                
                # Only save substantial chunks (skip tiny 5-word leftover fragments)
                if len(chunk) > 50: 
                    out_file.write(json.dumps({"text": chunk}, ensure_ascii=False) + '\n')
                    total_chunks += 1

    print(f"\nExtraction Complete!")
    print(f"Successfully created {total_chunks:,} pre-training chunks from {num_articles:,} articles.")

if __name__ == "__main__":
    format_wikipedia_data()