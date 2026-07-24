import os
import json
import re
from tqdm import tqdm
from datasets import load_dataset

def format_institutional_data(output_file="data/raw/institutional_formatted.jsonl", num_books=10000):
    """
    Downloads and formats the Institutional Books 1.0 dataset.
    Extracts post-processed OCR text, flattens page arrays, cleans unicode, 
    and chunks for Phase 1 pre-training.
    """

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping Book1 dataset download and formatting.")
        return

    print(f"Downloading {num_books:,} books from the Institutional Books dataset...")
    
    # We use streaming=True because this dataset contains nearly 1 million books!
    # Note: Adjust the dataset path if the actual Hugging Face ID differs.
    dataset = load_dataset("institutional/institutional-books-1.0", split="train", streaming=True, token=os.getenv("HF_TOKEN"))
    
    # Safely limit the stream to prevent runaway downloads
    dataset = dataset.take(num_books)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    chunk_size = 150 # Words per chunk (leaves room in the 256/512 context window)
    total_chunks = 0

    print(f"Flattening and chunking books to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for book in tqdm(dataset, total=num_books, desc="Parsing Books"):
            
            # 1. Handle the List Format and OCR Sources
            # We strongly prefer 'text_by_page_gen' (post-processed OCR) as it has fewer errors.
            # If it's missing, we fall back to 'text_by_page_src' (raw OCR).
            pages = book.get("text_by_page_gen") or book.get("text_by_page_src", [])
            
            if not pages:
                continue
            
            if isinstance(pages, list):
                raw_text = " ".join(pages)
            else:
                raw_text = str(pages) # Fallback just in case some rows are already strings
                
            # 2. Clean ALL unicode characters and OCR artifacts
            # This regex strips actual non-ASCII characters (covering all \u characters)
            clean_text = re.sub(r'[^\x00-\x7F]+', '', raw_text)
            
            # Catch any literal "\uXXXX" strings that might have escaped JSON parsing
            clean_text = re.sub(r'\\u[0-9a-fA-F]{4}', '', clean_text)
            
            # Fix any escaped quotes left over from the JSON formatting
            clean_text = re.sub(r"\\+(['\"])", r"\1", clean_text)
            
            # 3. Destroy newlines (\n) and normalize spaces
            # .split() with no arguments automatically strips \n, \t, and all extra spaces!
            clean_text = " ".join(clean_text.split())
            words = clean_text.split()
            
            # 4. Group into chunks of ~150 words for the context window
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                
                # Only save substantial chunks (skip 5-word leftover fragments)
                if len(chunk) > 50: 
                    out_file.write(json.dumps({"text": chunk}, ensure_ascii=False) + '\n')
                    total_chunks += 1

    print(f"\nExtraction Complete!")
    print(f"Successfully created {total_chunks:,} pre-training chunks from {num_books:,} academic books.")

if __name__ == "__main__":
    format_institutional_data()