import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer_blocks import TransformerModel
from dataset import CustomDataset

# --- Hyperparameters ---
# Assuming you upgraded to the 16GB Arc A770 as discussed!
vocab_size = 64000
d_model = 512
num_heads = 16
d_ff = 2048
num_layers = 6
max_len = 512
batch_size = 32

# Device setup (Intel Arc IPEX integration)
try:
    import intel_extension_for_pytorch as ipex
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Using Intel IPEX XPU backend on {device}")
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"IPEX not found. Falling back to standard backend: {device}")

def train_phase_1():
    print("\n" + "="*50)
    print("STARTING PHASE 1: PRE-TRAINING (World Knowledge)")
    print("="*50)
    
    dataset = CustomDataset("data/processed/phase1_pretraining.jsonl", "data/victorian_tokenizer.json", max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len).to(device)
    
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
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Phase 1 | Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
    print("Phase 1 Complete! Saving Base Model...")
    torch.save(model.state_dict(), "victorian_base_model.pt")
    return "victorian_base_model.pt"

def train_phase_2(base_model_path):
    print("\n" + "="*50)
    print("STARTING PHASE 2: INSTRUCTION FINE-TUNING (Chatbot Behavior)")
    print("="*50)
    
    # Load the Chat dataset
    dataset = CustomDataset("data/processed/phase2_finetuning.jsonl", "data/victorian_tokenizer.json", max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize a blank model and LOAD THE PHASE 1 WEIGHTS
    model = TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
    model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))
    model.to(device)
    
    # Lower learning rate so we don't destroy Phase 1 knowledge!
    learning_rate = 5e-5 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    if device.type == "xpu":
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.token_to_id("[UNK]"))
    
    # Just 1 Epoch to learn the tags without overfitting
    epochs = 1 
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Phase 2 | Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
    print("Phase 2 Complete! Saving Final Chatbot Model...")
    torch.save(model.state_dict(), "victorian_troll_final.pt")

if __name__ == "__main__":
    base_model_file = train_phase_1()
    train_phase_2(base_model_file)
    print("\nPIPELINE FINISHED. The 'victorian_troll_final.pt' file is ready for your API!")