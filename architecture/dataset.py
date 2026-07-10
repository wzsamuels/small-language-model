import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

class CustomDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer_file, max_length=512):
        # 1. Load the tokenizer and initialize the dataset
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.max_length = max_length
        self.data = []
        
        # Load your JSONL file
        print(f"Loading data from {jsonl_file}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
                    
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        # Extract the text for this specific conversation exchange
        messages = self.data[idx].get("messages", [])
        
        # Combine the system, user, and assistant text into one string for training
        
        raw_text = " ".join([f"<{m['role']}> {m['content']}" for m in messages])

        # 2. Use your custom tokenizer to convert the human text into sequences of integer IDs
        encoded = self.tokenizer.encode(raw_text)
        
        # 3. Chop those sequences into fixed-length "context windows" (e.g., 512 tokens)
        ids = encoded.ids[:self.max_length]
        
        # Pad sequences that are shorter than max_length using the [UNK] token
        padding_length = self.max_length - len(ids)
        if padding_length > 0:
            pad_id = self.tokenizer.token_to_id("[UNK]")
            ids.extend([pad_id] * padding_length)
            
        # 4. Package them into PyTorch Tensors
        # X is the input sequence, Y is the target sequence (shifted by one token)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        
        return x, y

# Initialize the Dataset
#dataset = VictorianTrollDataset(
#    jsonl_file='data/master_training_data.jsonl', 
#    tokenizer_file='data/victorian_troll_tokenizer.json',
#    max_length=512
#)

# Initialize the DataLoader to feed the PyTorch Tensors to your GPU in batches
#dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

# Test it out to see the shape of your data
#x_batch, y_batch = next(iter(dataloader))
#print(f"Batch X shape: {x_batch.shape}")
#print(f"Batch Y shape: {y_batch.shape}")
