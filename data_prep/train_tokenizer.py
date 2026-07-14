from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import json
import os

def train_custom_tokenizer(dataset_path, output_path, vocab_size=32000):

    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping tokenizer phase.")
        return

    print(f"Initializing BPE Tokenizer...")
    
    # 1. Initialize a blank BPE Tokenizer
    # [UNK] is the token used if the model encounters a character it has literally never seen before
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # 2. Pre-tokenize by splitting on whitespace
    # This prevents the tokenizer from trying to merge across word boundaries (e.g., merging the end of one word with the start of the next)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 3. Define the Trainer
    # We define our target vocabulary size and our special control tokens
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    )

    print(f"Extracting raw text from {dataset_path}...")
    
    # 4. Create a generator to feed the text to the tokenizer efficiently
    # The tokenizer needs raw strings, not JSON syntax, so we extract just the "content" values
    def batch_iterator():
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                for message in data.get("messages", []):
                    yield message["content"]

    # 5. Train! (This will take a few seconds to a minute depending on your CPU)
    print("Training in progress (calculating character frequencies and merging)...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    tokenizer.save(output_path)
    print(f"Success! Tokenizer saved to {output_path}")
    
    # Test it out to see the results
    #test_string = "You are a Victorian-era troll. Wherefore art thou posting cringe?"
    #encoded = tokenizer.encode(test_string)
    #print(f"\nTest Encoding:")
    #print(f"Original: {test_string}")
    #print(f"Tokens: {encoded.tokens}")
    #print(f"IDs: {encoded.ids}")

# Execute
# train_custom_tokenizer('../data/master_training_data.jsonl', vocab_size=32000)

if __name__ == "__main__":
    # Test it out to see the results
    test_string = "You are a Victorian-era troll. Wherefore art thou posting cringe?"
    
    encoded = tokenizer.encode(test_string)
    print(f"\nTest Encoding:")
    print(f"Original: {test_string}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")