from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from architecture.transformer_blocks import TransformerModel
from tokenizers import Tokenizer
from config import hyperparams

# 1. Initialize FastAPI
app = FastAPI(title="Victorian Troll API")

# 2. Setup Device & Load Model (Global scope so it only loads once)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

tokenizer = Tokenizer.from_file("data/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

model = TransformerModel(vocab_size=vocab_size, d_ff=hyperparams["d_ff"], d_model=hyperparams["d_model"], num_heads=hyperparams["num_heads"], num_layers=hyperparams["num_layers"], max_len=hyperparams["max_len"])
model.load_state_dict(torch.load("custom_model.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()

# 3. Define the Request Data Structure
class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.9
    max_new_tokens: int = 50

# 4. The Generation Endpoint
@app.post("/generate")
async def generate_response(request: ChatRequest):
    try:
        # Format the prompt with your required special tags
        full_prompt = f"<|system|> You are an anarchist British punk rocker. <|user|> {request.prompt} <|assistant|>"
        
        # Tokenize
        input_ids = tokenizer.encode(full_prompt).ids
        x = torch.tensor([input_ids], dtype=torch.long).to(device)

        prompt_length = x.size(1)

        # Autoregressive Generation Loop (simplified for the API)
        with torch.no_grad():
            for _ in range(request.max_new_tokens):
                logits = model(x)
                next_token_logits = logits[0, -1, :] / request.temperature
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
                
                if next_token.item() == tokenizer.token_to_id("<|end|>"):
                    break

        # Decode and scrub special tags
        generated_ids = x[0, prompt_length:].tolist()
        
        # Decode only the new text
        clean_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        return {"response": clean_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally using: uvicorn api:app --host 127.0.0.1 --port 8000