import os
import random
import json
from tqdm import tqdm

def blend_finetuning_data(output_file="data/processed/phase2_finetuning.jsonl"):
    """
    Interleaves our structured conversational datasets (SODA, OASST, Gutenberg Plays)
    into a single shuffled file for Phase 2 Fine-Tuning.
    """
    input_files = [
        "data/raw/soda_formatted.jsonl",
        "data/raw/oasst_formatted.jsonl",
        "data/raw/gutenberg_formatted.jsonl"
    ]
    
    active_files = []
    for filepath in input_files:
        if os.path.exists(filepath):
            print(f"Sizing up {filepath}...")
            active_files.append(open(filepath, 'r', encoding='utf-8'))
        else:
            print(f"Warning: {filepath} not found. Skipping.")

    if not active_files:
        print("CRITICAL: No input files found to blend!")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\nBlending {len(active_files)} chat datasets into {output_file}...")
    
    lines_written = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        with tqdm(desc="Blending Chat Rows") as pbar:
            while active_files:
                file_handle = random.choice(active_files)
                line = file_handle.readline()
                
                if not line:
                    file_handle.close()
                    active_files.remove(file_handle)
                else:
                    out_f.write(line)
                    lines_written += 1
                    pbar.update(1)

    print(f"\nSuccess! Blended {lines_written:,} conversation threads for Phase 2 Fine-Tuning.")

if __name__ == "__main__":
    blend_finetuning_data()