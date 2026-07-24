import os
import random
from tqdm import tqdm

def blend_pretraining_data(output_file="data/processed/phase1_pretraining.jsonl"):
    """
    Interleaves multiple massive JSONL files into a single heterogeneous dataset 
    without overloading system RAM.
    """
    input_files = [
        "data/raw/pg19_formatted.jsonl",
        "data/raw/wikipedia_formatted.jsonl",
        "data/raw/openwebtext_formatted.jsonl"
    ]
    
    # 1. Verify files exist and open them
    active_files = []
    for filepath in input_files:
        if os.path.exists(filepath):
            # Calculate rough line count for the progress bar (optional but helpful)
            print(f"Sizing up {filepath}...")
            # We open the file handle and keep it open
            active_files.append(open(filepath, 'r', encoding='utf-8'))
        else:
            print(f"Warning: {filepath} not found. Skipping.")

    if not active_files:
        print("CRITICAL: No input files found to blend!")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nBlending {len(active_files)} datasets into {output_file}...")
    
    lines_written = 0
    
    # 2. Randomly pluck lines from the open files until they are all empty
    with open(output_file, 'w', encoding='utf-8') as out_f:
        with tqdm(desc="Blending Rows") as pbar:
            while active_files:
                # Pick a random open file
                file_handle = random.choice(active_files)
                
                # Read a single line
                line = file_handle.readline()
                
                if not line:
                    # If the file is empty, close it and remove it from our active list
                    file_handle.close()
                    active_files.remove(file_handle)
                else:
                    # Write the line to our master file
                    out_f.write(line)
                    lines_written += 1
                    pbar.update(1)

    print(f"\nSuccess! Blended {lines_written:,} total chunks into Phase 1 Pre-Training data.")

if __name__ == "__main__":
    blend_pretraining_data()