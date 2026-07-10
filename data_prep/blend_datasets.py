import json
import random
import math

def blend_data(gutenberg_path, modern_path, output_path, gutenberg_weight=0.75):
    print("Loading datasets into memory")

    with open(gutenberg_path, 'r', encoding='utf-8') as f:
        gutenberg_data = [line for line in f if line.strip()]

    with open(modern_path, 'r', encoding='utf-8') as f:
        modern_data = [line for line in f if line.strip()]

    g_count = len(gutenberg_data)
    m_count = len(modern_data)
    
    print(f"Original Gutenberg Rows: {g_count}")
    print(f"Original Modern Rows: {m_count}")

    # Calculate the required number of modern rows to hit the target ratio
    # If Gutenberg is 75%, then Modern is 25%. 
    # Target Modern = (Gutenberg Count * Modern Weight) / Gutenberg Weight
    modern_weight = 1.0 - gutenberg_weight
    target_modern_count = math.ceil((g_count * modern_weight) / gutenberg_weight)
    
    print(f"Targeting a {gutenberg_weight*100:.0f}/{modern_weight*100:.0f} split.")
    print(f"Oversampling modern data to {target_modern_count} rows...")

    # Oversample (duplicate) the modern data to hit the target count
    multiplier = target_modern_count // m_count
    remainder = target_modern_count % m_count
    
    oversampled_modern = (modern_data * multiplier) + random.sample(modern_data, remainder)
    
    # Combine the datasets
    master_dataset = gutenberg_data + oversampled_modern
    
    print(f"Total rows in blended dataset: {len(master_dataset)}")
    print("Shuffling the master dataset...")
    
    # Shuffle aggressively so the model doesn't learn in chunks
    random.shuffle(master_dataset)
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in master_dataset:
            f.write(line)
            
    print("Blending complete! Your data is ready.")

# Execute the script
# Change gutenberg_weight to adjust the ratio (e.g., 0.50 for a 50/50 split)
#blend_and_shuffle(
#    gutenberg_path='../data/training_data.jsonl', 
#    modern_path='../data/modern_training_data.jsonl', 
#    output_path='../data/master_training_data_75.jsonl',
#    gutenberg_weight=0.75 
#)
#
#
#blend_and_shuffle(
#    gutenberg_path='../data/training_data.jsonl', 
#    modern_path='../data/modern_training_data.jsonl', 
#    output_path='../data/master_training_data_50.jsonl',
#    gutenberg_weight=0.50 
#)

#blend_and_shuffle(
#    gutenberg_path='../data/training_data.jsonl', 
#    modern_path='../data/modern_training_data.jsonl', 
#    output_path='../data/master_training_data_25.jsonl',
#    gutenberg_weight=0.25 
#)
