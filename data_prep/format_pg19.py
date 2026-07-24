import os
import json
from tqdm import tqdm
from datasets import load_dataset

def format_pg19_data(output_file="data/raw/pg19_formatted.jsonl", num_books=5000):
    """
    Downloads DeepMind's PG-19 (Project Gutenberg) dataset from Hugging Face 
    and chunks the books for Phase 1 pre-training.
    """
    print(f"Downloading {num_books:,} books from the PG-19 dataset...")
    
    # PG-19 is around 11GB of pure text, so we use streaming=True
    dataset = load_dataset("pg19", split="train", streaming=True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    chunk_size = 150 # Words per chunk to fit your 256 context window
    total_chunks = 0

    print(f"Chunking books and saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for idx, book in enumerate(tqdm(dataset, total=num_books, desc="Parsing Books")):
            if idx >= num_books:
                break
                
            # PG-19 stores the book content in the "text" key
            raw_text = book.get("text", "")
            if not raw_text:
                continue
                
            # DeepMind already removed the Gutenberg headers/footers, 
            # so we just need to clean up basic whitespace
            clean_text = " ".join(raw_text.split())
            words = clean_text.split()
            
            # Group into chunks of ~150 words
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                
                # Save substantial chunks
                if len(chunk) > 50: 
                    out_file.write(json.dumps({"text": chunk}, ensure_ascii=False) + '\n')
                    total_chunks += 1

    print(f"\nExtraction Complete!")
    print(f"Successfully created {total_chunks:,} pre-training chunks from {num_books:,} books.")

if __name__ == "__main__":
    format_pg19_data()