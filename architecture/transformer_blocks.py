import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # PyTorch's built-in lookup table
        # It creates a matrix of shape (vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x is our batch of token IDs from the DataLoader (e.g., shape: [8, 511])
        # The embedding layer converts this to shape: [8, 511, d_model]
        
        # In the original Transformer paper ("Attention Is All You Need"), 
        # embeddings are scaled by the square root of the embedding dimension.
        # This helps stabilize the variance of the embeddings before adding positional encoding.
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model) to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create a vector of shape (max_len, 1) with positions (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for the sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices in the matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the matrix
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as a buffer so PyTorch knows this isn't a trainable parameter (we don't want it to learn), 
        # but it should be saved and moved to the GPU along with the rest of the model.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # We slice self.pe to match the actual sequence length of x, just in case it's shorter than max_len
        # Then we add the positional encoding directly to the word embeddings
        x = x + self.pe[:, :x.size(1), :]
        
        # Apply dropout for regularization
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # d_model must be perfectly divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        
        # Linear projections for Queries, Keys, and Values
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Final output projection
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. Project Q, K, V and split into multiple heads
        # Shape transforms from [batch_size, seq_len, d_model] 
        # to [batch_size, num_heads, seq_len, d_k]
        query = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention: (Q * K^T) / sqrt(d_k)
        # Calculates how much focus each word should give to every other word
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply the mask (critical for hiding future words during text generation or ignoring padding)
        if mask is not None:
            # We fill masked positions with a massive negative number so Softmax turns them to 0
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        # 3. Apply Softmax to convert scores into percentages/probabilities
        attention_weights = torch.softmax(attention_scores, dim=-1)
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)
            
        # 4. Multiply by Values to get the context-aware word representations
        attention_output = attention_weights @ value
        
        # 5. Concatenate the heads back together
        # Transforms back to [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 6. Pass through the final linear layer
        return self.w_o(attention_output)

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # A simple 2-layer neural network
        # d_ff is usually 4 times the d_model (e.g., 256 -> 1024)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        
        # Layer Normalization stabilizes training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Self-Attention with a "Residual Connection" (adding x back to the output)
        attention_out = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_out))
        
        # 2. Feed-Forward with another Residual Connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed = InputEmbeddings(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        
        # Create a stack of repeating Decoder Blocks
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        # Final projection layer to convert embeddings back into vocabulary probabilities
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # 1. Convert IDs to embeddings and add positional encoding
        x = self.embed(x)
        x = self.pos_enc(x)
        
        # 2. Pass through all the Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
            
        # 3. Project to vocab size to get the raw scores (logits) for the next word
        return self.linear(x)

# --- Test the Architecture ---
if __name__ == "__main__":
    print("Testing Transformer Architecture...")
    
    vocab_size = 32000
    d_model = 256
    num_heads = 8
    d_ff = 1024      # Generally 4x d_model
    num_layers = 4   # 4 repeating blocks for a "micro" model
    max_len = 512
    batch_size = 8
    seq_len = 511
    
    # Initialize the complete model
    model = TransformerModel(vocab_size, d_model, num_heads, d_ff, num_layers, max_len)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters.")
    
    # Create a dummy batch of token IDs (e.g., 8 conversations, 511 tokens each)
    x_batch = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input batch shape: {x_batch.shape}")
    
    # Run the forward pass
    print("Running forward pass...")
    logits = model(x_batch)
    print(f"Final output shape: {logits.shape}")  # Expected: [8, 511, 32000]
    
    # Verify the output shape
    if logits.shape == (batch_size, seq_len, vocab_size):
        print("✅ Test passed successfully! The architecture is outputting the correct tensor shape.")
    else:
        print("❌ Test failed! Unexpected output shape.")
