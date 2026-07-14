import torch
import random
import torch.nn.functional as F
from tokenizers import Tokenizer
from architecture.transformer_blocks import TransformerModel
from config import persona

def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, device, max_len):
    # Set the model to evaluation mode (disables dropout layers)
    model.eval() 
    
    # Convert our text prompt into integer IDs
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"\nFormatted Prompt: {prompt}")
    print("Bot: ", end="")
    
    # Track only the newly generated tokens
    generated_new_ids = []
    
    # The Autoregressive Loop
    with torch.no_grad(): # Tell PyTorch not to track gradients (saves memory)
        for _ in range(max_new_tokens):
            # Pass the current sequence through the model
            logits = model(input_ids)
            
            # Get the predictions for the very last token in the sequence
            next_token_logits = logits[0, -1, :]
            
            # Apply a repetition penalty of 1.2 to previously generated words
            repetition_penalty = 1.2
            for past_token_id in set(generated_new_ids):
                if next_token_logits[past_token_id] > 0:
                    next_token_logits[past_token_id] /= repetition_penalty
                else:
                    next_token_logits[past_token_id] *= repetition_penalty

            # Apply Temperature
            next_token_logits = next_token_logits / temperature
            
            # Convert to probabilities and sample the next word
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Keep a running list of just the new IDs
            generated_new_ids.append(next_token_id)
            
            # Decode the integer ID back into a human-readable word and print it
            # skip_special_tokens=False ensures we can actually see the tags if they generate
            word = tokenizer.decode([next_token_id], skip_special_tokens=False)
            print(word, end="", flush=True)
            
            # We decode the entire newly generated sequence to check for structural tags
            new_text = tokenizer.decode(generated_new_ids, skip_special_tokens=False)
            
            # If the bot tries to end the conversation, or tries to speak for the user, stop immediately!
            if "<|end|>" in new_text or "<|user|>" in new_text or "<assistant" in new_text or "<|assistant" in new_text:
                print("\n[Turn Ended]")
                break
            
            # Append the new token to the sequence for the next loop iteration
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            
            # Safety check: don't exceed the model's max context length
            if input_ids.size(1) >= max_len:
                break
                
    print("\n")                

def run_test():
    # Force CPU since you are testing on your laptop
    device = torch.device("cuda")
    print(f"Loading model on {device}...")
    
    tokenizer = Tokenizer.from_file('data/tokenizer.json')
    
    # Re-create the architecture using the EXACT SAME hyperparameters as train.py
    vocab_size = 32000
    d_model = 256
    num_heads = 8
    d_ff = 1024
    num_layers = 4
    max_len = 512
    
    model = TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
    
    # Load the saved weights into the architecture
    model.load_state_dict(torch.load("custom_model.pt", map_location=device, weights_only=True))
    model.to(device)
    
    while True:
        # Format the prompt EXACTLY how it looked in the training data JSONL
        system_prompt = persona
        user_message = input("> ")

        if user_message == "quit":
            break
        
        # Wrap it in the ChatML tags we used in the DataLoader
        formatted_prompt = f"<|system|> {system_prompt} <|user|> {user_message} <|assistant|>"
        
        # Generate! Try adjusting the temperature between 0.5 (boring) and 1.2 (insane)
        generate_text(model, tokenizer, formatted_prompt, max_new_tokens=50, temperature=0.5, device=device, max_len=max_len)

if __name__ == "__main__":
    run_test()
