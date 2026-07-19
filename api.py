from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from architecture.transformer_blocks import TransformerModel
from tokenizers import Tokenizer
from config import hyperparms, persona

app = FastAPI(title="SLM API")

# 2. Setup Device & Load Model (Global scope so it only loads once)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Loading model on {device}...")

tokenizer = Tokenizer.from_file("models/tokenizer-v1.json")
vocab_size = tokenizer.get_vocab_size()

model = TransformerModel(vocab_size=hyperparms["vocab_size"], d_ff=hyperparms["d_ff"], d_model=hyperparms["d_model"], num_heads=hyperparms["num_heads"], num_layers=hyperparms["num_layers"], max_len=hyperparms["max_len"])
model.load_state_dict(torch.load("models/custom_model-v1.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()

# 3. Define the Request Data Structure
class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 1
    max_new_tokens: int = 50

def api_generate_text(prompt: str, max_new_tokens: int, temperature: float):
    # Convert text to IDs
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long).unsqueeze(0).to(device)
    
    generated_new_ids = []
    new_text = ""
    
    # torch.no_grad() is crucial here! It prevents RAM spikes on your 2GB server.
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
            
            if input_ids.size(1) >= hyperparms["max_len"]:
                break
                
    # API CHANGE: Return the string instead of printing it!
    return new_text.strip()

@app.post("/generate")
async def generate_response(request: ChatRequest):
    try:
        # Format the prompt with your required special tags
        full_prompt = f"<|system|> {persona} <|user|> {request.prompt} <|assistant|>"
        
        response_text = api_generate_text(
            prompt=full_prompt, 
            max_new_tokens=request.max_new_tokens, 
            temperature=request.temperature
        )
        
        return {"response": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally using: uvicorn api:app --host 127.0.0.1 --port 8000