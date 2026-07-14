import os
import re
import json
from tqdm import tqdm

def clean_gutenberg_text(text):
    """Removes Gutenberg-specific formatting artifacts."""
    text = re.sub(r'\[\d+\]', '', text) # Remove footnotes
    text = re.sub(r'\[.*?\]', '', text) # Remove brackets
    text = text.replace('_', '')        # Remove underscores
    text = re.sub(r'\s+', ' ', text).strip() # Normalize spaces
    return text

def format_gutenberg(input_dir, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found. Skipping novel extraction.")
        return

    novel_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    success_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for filename in tqdm(novel_files, desc="Parsing Novels"):
            filepath = os.path.join(input_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as file:
                raw_text = file.read()
                
            # Extract just the book content
            start_match = re.search(r'\*\*\* START OF THE PROJECT GUTENBERG.*?\*\*\*', raw_text)
            end_match = re.search(r'\*\*\* END OF THE PROJECT GUTENBERG.*?\*\*\*', raw_text)
            
            if start_match and end_match:
                book_text = raw_text[start_match.end():end_match.start()]
            else:
                continue # Skip if we can't find the boundaries
                
            # Clean the text
            clean_text = clean_gutenberg_text(book_text)
            
            # Split into words so we can chunk it
            words = clean_text.split()
            
            # Group into chunks of ~150 words (leaves room for the tokenizer)
            chunk_size = 150
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                
                # Notice we use "text" instead of "messages"!
                if len(chunk) > 50: # Only save substantial chunks
                    out_file.write(json.dumps({"text": chunk}, ensure_ascii=False) + '\n')
            
            success_count += 1

    print(f"Successfully chunked {success_count} novels for pre-training!")

if __name__ == "__main__":
    format_gutenberg(input_dir="data/raw/gutenberg_all", output_file="data/processed/gutenberg_formatted.jsonl")