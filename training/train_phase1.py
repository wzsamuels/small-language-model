import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from architecture.transformer_blocks import TransformerModel
from architecture.dataset import CustomDataset
from config import hyperparms

if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Training on native backend: {device}")

def train_phase_1():
    print("\n" + "="*50)
    print("STARTING PHASE 1: PRE-TRAINING (World Knowledge)")
    print("="*50)
    
    dataset = CustomDataset("data/processed/phase1_pretraining.jsonl", "models/tokenizer-v3.json")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = TransformerModel(hyperparms["vocab_size"], hyperparms["d_model"],
                            hyperparms["num_heads"], hyperparms["d_ff"], hyperparms["num_layers"],
                            hyperparms["max_len"]).to(device)
    
    # Standard learning rate for training from scratch
    learning_rate = 3e-4 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    if device.type == "xpu":
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.token_to_id("[UNK]"))
    
    # 2 Epochs for the massive Pre-Training corpus
    epochs = 2 
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            loss = criterion(logits.view(-1, hyperparms["vocab_size"]), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Phase 1 | Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
    print("Phase 1 Complete! Saving Base Model...")
    torch.save(model.state_dict(), "models/model-v3.pt")
    return "victorian_base_model.pt"

if __name__ == "__main__":
    train_phase_1()
