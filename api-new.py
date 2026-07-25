import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List # NEW: Import List for our message arrays
from tokenizers import Tokenizer
from transformer_blocks import TransformerModel

# Initialize the API app
app = FastAPI()

# 1. Global State: Load the model and tokenizer ONCE when the server starts
device = torch.device("cpu") # AWS standard instances don't have GPUs
tokenizer = Tokenizer.from_file('victorian_troll_tokenizer.json')

vocab_size = 32000
d_model = 256
num_heads = 8
d_ff = 1024
num_layers = 4
max_len = 128

model = TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
model.load_state_dict(torch.load("victorian_troll_model.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()

# 2. Define the expected incoming JSON payload for conversation history
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = 50
    temperature: float = 0.9

# 3. Context Window Management
def manage_context_window(messages_list, max_len, max_new_tokens):
    """Prunes old messages if the conversation exceeds the model's memory."""
    max_prompt_len = max_len - max_new_tokens
    
    while True:
        # Build the ChatML string using our tags
        prompt_string = " ".join([f"<|{m.role}|> {m.content}" for m in messages_list])
        
        # Append the assistant tag to prompt the model to reply
        prompt_string += " <|assistant|>"
        
        # Count the tokens
        token_count = len(tokenizer.encode(prompt_string).ids)
        
        if token_count <= max_prompt_len:
            return prompt_string
            
        # We want to keep the System prompt (index 0) and the newest user prompt.
        # If we are down to just those two and it's still too long, we have to stop.
        if len(messages_list) <= 2:
            print("Warning: Input is too long for the context window!")
            return prompt_string
            
        # Pop the oldest conversational turn (index 1)
        messages_list.pop(1)

# 4. The Adapted Generation Function
def api_generate_text(messages: List[Message], max_new_tokens: int, temperature: float):
    # Format the history and prune if necessary
    prompt = manage_context_window(messages, max_len, max_new_tokens)
    
    # Convert text to IDs
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)
    
    generated_new_ids = []
    new_text = ""
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            # Repetition Penalty
            repetition_penalty = 1.2
            for past_token_id in set(input_ids[0].tolist()):
                if next_token_logits[past_token_id] > 0:
                    next_token_logits[past_token_id] /= repetition_penalty
                else:
                    next_token_logits[past_token_id] *= repetition_penalty

            # Temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-K Sampling
            top_k = 40
            top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
            min_top_k_value = top_k_values[-1]
            next_token_logits[next_token_logits < min_top_k_value] = float('-inf')
            
            # Sample next word
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            generated_new_ids.append(next_token_id)
            new_text = tokenizer.decode(generated_new_ids, skip_special_tokens=False)
            
            # Stop Checking
            stop_sequences = ["<|end|>", "<|user|>", "<assistant", "<|assistant"]
            hit_stop = False
            for seq in stop_sequences:
                if seq in new_text:
                    new_text = new_text.split(seq)[0]
                    hit_stop = True
                    break
            
            if hit_stop:
                break
            
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            
            if input_ids.size(1) >= max_len:
                break
                
    # Return the generated string
    return new_text.strip()

# 5. The API Endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Call the adapted function and pass the message history
    response_text = api_generate_text(
        messages=request.messages, 
        max_new_tokens=request.max_new_tokens, 
        temperature=request.temperature
    )
    
    return {"reply": response_text}



    {
  "messages": [
    {"role": "system", "content": "You are an anarchist British punk rocker."},
    {"role": "user", "content": "Oi, what's your name?"},
    {"role": "assistant", "content": "None of your business, mate."},
    {"role": "user", "content": "Why are you so rude?"}
  ],
  "temperature": 0.9,
  "max_new_tokens": 50
}

The FastAPI server will automatically ingest that JSON array, prune out the oldest turns if the history exceeds your 256 or 512 `max_len`, cap it with an `<|assistant|>` tag, and fire off the neural network!