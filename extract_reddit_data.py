import json
import random
from datasets import load_dataset

def format_social_data(output_filepath, sample_size=5000):
    print("Downloading SODA dataset from Hugging Face...")
    # 'allenai/soda' is a highly reliable dataset of everyday social chitchat
    dataset = load_dataset("allenai/soda", split="train")
    
    # Shuffle the dataset and take a subset so we don't overwhelm your mix
    dataset = dataset.shuffle(seed=42).select(range(sample_size))
    
    personas = [
        "You are a Victorian-era troll.",
        "You are an angry Victorian ghost.",
        "You are a polite but extremely condescending scholar.",
        "You are a time-traveler who is very confused by modern technology."
    ]
    
    print(f"Formatting {sample_size} conversations...")
    
    with open(output_filepath, 'w', encoding='utf-8') as out_file:
        for row in dataset:
            # SODA stores the conversation as a simple list of strings in the 'dialogue' column
            dialogue = row.get('dialogue', [])
            
            # We need at least a back-and-forth exchange (2 turns)
            if not dialogue or len(dialogue) < 2:
                continue
                
            user_text = dialogue[0]
            assistant_text = dialogue[1]
            
            assigned_persona = random.choice(personas)
            
            chat_entry = {
                "messages": [
                    {"role": "system", "content": assigned_persona},
                    {"role": "user", "content": str(user_text).strip()},
                    {"role": "assistant", "content": str(assistant_text).strip()}
                ]
            }
            out_file.write(json.dumps(chat_entry) + '\n')

    print(f"Successfully saved conversational data to {output_filepath}")

# Execute the extraction
format_social_data('modern_training_data.jsonl', sample_size=300_000)
