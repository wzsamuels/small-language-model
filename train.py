import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Import our custom classes from the other files
# (Ensure your previous Dataset code is saved as dataset.py)
from architecture.dataset import CustomDataset 
# (Ensure the architecture code from the Canvas is saved as transformer_blocks.py)
from architecture.transformer_blocks import TransformerModel 
from config import hyperparms

def train_model(input_training_file, input_tokenizer_file, output_file, device="cuda"):
    # 1. Hyperparameters
    vocab_size = hyperparms["vocab_size"]
    d_model = hyperparms["d_model"]
    num_heads = hyperparms["num_heads"]
    d_ff = hyperparms["d_ff"]
    num_layers = hyperparms["num_layers"]
    batch_size = hyperparms["batch_size"]
    epochs = hyperparms["epochs"]
    learning_rate = hyperparms["learning_rate"] # Standard starting learning rate for Transformers
    max_len = hyperparms["max_len"]

    # Detect if a GPU is available, otherwise fall back to CPU
    device = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")

    print(f"Training on device: {device}")

    # 2. Load Data
    print("Loading dataset and tokenizer...")
    dataset = CustomDataset(
        jsonl_file=input_training_file, 
        tokenizer_file=input_tokenizer_file
    )
    
    print(f"Dataset loaded with {len(dataset)} samples.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. Initialize Model
    print("Initializing Transformer architecture...")
    model = TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
    model.to(device) # Move the model to the GPU

    # 4. Optimizer and Loss Function
    # AdamW is the industry standard optimizer for Transformers (handles weight decay well)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # CrossEntropyLoss compares the model's predictions to the actual next word.
    # We use ignore_index to tell the model not to calculate loss on our [UNK] padding tokens,
    # assuming your tokenizer mapped [UNK] to ID 0.
    pad_token_id = dataset.tokenizer.token_to_id("[UNK]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # 5. The Training Loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train() # Set model to training mode (enables dropout)
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            # Move the input (x) and target (y) tensors to the GPU
            x, y = x.to(device), y.to(device)
            
            # Clear old gradients from the last step
            optimizer.zero_grad()
            
            # Forward pass: get the model's predictions
            logits = model(x)
            
            # Reshape for CrossEntropyLoss:
            # PyTorch expects logits as a 2D grid [batch_size * seq_len, vocab_size]
            # and targets as a 1D list [batch_size * seq_len]
            logits_reshaped = logits.view(-1, vocab_size)
            y_reshaped = y.view(-1)
            
            # Calculate how far off the predictions were
            loss = criterion(logits_reshaped, y_reshaped)
            
            # Backward pass: calculate the gradients (the direction to adjust the weights)
            loss.backward()
            
            # Optimizer step: actually update the neural network's weights
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print an update every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        # End of epoch summary
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---")

    # 6. Save the Trained Model
    output_path = "custom_model.pt"
    torch.save(model.state_dict(), output_path)
    print(f"Training complete! Model weights saved to {output_path}")

if __name__ == "__main__":
    train_model()
