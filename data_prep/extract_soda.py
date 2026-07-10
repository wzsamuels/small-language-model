import json
import random
import os
from datasets import load_dataset
from tqdm import tqdm
from config import persona

def format_soda_data(input_file="data/raw/soda_raw.jsonl", output_file="data/raw/soda_formatted.jsonl"):
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping soda formatting.")
        return

    if not os.path.exists(input_file):
        pass

    soda_data = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Loading soda data from file"):
            row = json.loads(line)
            soda_data.append(row)
    
    print(f"Successfully loaded {len(soda_data)} rows into memory!")
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for row in tqdm(soda_data, desc="Formatting conversations"):
            # SODA stores the conversation as a simple list of strings in the 'dialogue' column
            dialogue = row.get('dialogue', [])
            
            # We need at least a back-and-forth exchange (2 turns)
            if not dialogue or len(dialogue) < 2:
                continue
                
            user_text = dialogue[0]
            assistant_text = dialogue[1]          
            
            chat_entry = {
                "messages": [
                    {"role": "system", "content": persona},
                    {"role": "user", "content": str(user_text).strip()},
                    {"role": "assistant", "content": str(assistant_text).strip()}
                ]
            }
            out_file.write(json.dumps(chat_entry) + '\n')

    print(f"Successfully saved soda data to {output_file}")

# Execute the extraction
if __name__ == "__main__":
    format_soda_data()
