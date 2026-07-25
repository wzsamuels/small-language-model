import torch

def inspect_model(pt_file_path):
    print(f"--- Cracking open {pt_file_path} ---")
    
    # Load the weights dictionary into CPU memory
    state_dict = torch.load(pt_file_path, map_location="cpu", weights_only=True)
    
    print("\nLayer Name | Tensor Shape")
    print("-" * 50)
    
    # Loop through every layer in the model and print its physical dimensions
    for layer_name, tensor in state_dict.items():
        # We only print a few key layers so the terminal isn't flooded
        if "embed" in layer_name or "pos_enc" in layer_name or "layers.0" in layer_name or "linear" in layer_name:
            print(f"{layer_name}: {list(tensor.shape)}")
            
    print("-" * 50)
    
    # Let's count the number of layers dynamically
    layer_indices = [int(key.split('.')[1]) for key in state_dict.keys() if key.startswith('layers.')]
    if layer_indices:
        num_layers = max(layer_indices) + 1
        print(f"\nTotal Decoder Layers found: {num_layers}")

if __name__ == "__main__":
    inspect_model(pt_file_path="models/custom_model-v1.pt")