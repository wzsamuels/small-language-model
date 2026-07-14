from architecture.dataset import CustomDataset

def run_preflight_check(jsonl_file, tokenizer_file):
    print("=== Starting Pre-Flight Check ===\n")
    
    # 1. Initialize the dataset exactly as train.py will
    dataset = CustomDataset(
        jsonl_file=jsonl_file, 
        tokenizer_file=tokenizer_file,
        max_length=512
    )
    
    print(f"\nSuccessfully loaded {len(dataset):,} conversations.")
    print("Pulling the first conversation for inspection...\n")
    
    # 2. Grab the very first tensor pair from the dataset
    x, y = dataset[0]
    
    # 3. Convert the PyTorch tensor back to a list of Python integers
    input_ids = x.tolist()
    
    # 4. Decode the IDs back into text
    # Setting skip_special_tokens=False ensures we can see [UNK], <|user|>, etc.
    decoded_text = dataset.tokenizer.decode(input_ids, skip_special_tokens=False)
    
    print("--------------------------------------------------")
    print("EXACT STRING FED TO THE NEURAL NETWORK:")
    print("--------------------------------------------------")
    print(decoded_text)
    print("--------------------------------------------------\n")
    
    # 5. Automated Sanity Checks
    print("Automated Tag Checks:")
    tags = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    for tag in tags:
        if tag in decoded_text:
            print(f"✅ Found '{tag}'")
        else:
            print(f"❌ MISSING '{tag}' - Check your formatting scripts!")

if __name__ == "__main__":
    # Adjust these paths if your master files are stored elsewhere!
    run_preflight_check(
        jsonl_file='data/processed/master_training_data.jsonl',
        tokenizer_file='data/tokenizer.json'
    )