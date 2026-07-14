import json
import random
import os
from datasets import load_dataset
from tqdm import tqdm

persona = "You are an anarchist British punk rocker."

def format_soda_data(input_file, output_file):
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping soda formatting.")
        return

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"CRITICAL: Cannot find {input_file}. Did the download phase complete successfully?")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    formatted_threads = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Processing SODA dialogues"):
            row = json.loads(line)
            dialogue_list = row.get("dialogue", [])
            
            # Skip empty or broken conversations
            if not dialogue_list or len(dialogue_list) < 2:
                continue
                
            thread = [{"role": "system", "content": persona}]
            
            # 2. Loop through the ENTIRE dialogue array
            for idx, turn_text in enumerate(dialogue_list):
                # SODA alternates speakers. Even indices (0, 2, 4...) are Speaker 1 (User).
                # Odd indices (1, 3, 5...) are Speaker 2 (Assistant).
                role = "user" if idx % 2 == 0 else "assistant"
                
                thread.append({
                    "role": role,
                    "content": turn_text
                })
                
            formatted_threads.append({"messages": thread})

    print(f"Successfully processed {len(formatted_threads):,} multi-turn SODA conversations!")
    print(f"Saving formatted data to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for thread in tqdm(formatted_threads, desc="Saving threads"):
            f.write(json.dumps(thread, ensure_ascii=False) + '\n')

    print("SODA formatting complete!")

# Execute the extraction
if __name__ == "__main__":
    format_soda_data(input_file="data/raw/soda_raw.jsonl", output_file="data/processed/soda_formatted.jsonl")
