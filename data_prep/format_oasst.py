import json
import os
from tqdm import tqdm

persona = "You are an anarchist British punk rocker."

def format_oasst_data(input_file, output_file):
    """Reconstructs linear conversations from the OASST message trees."""
    print("Loading raw OpenAssistant data into memory...")
    messages = {}
    
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping OASST formatting.")
        return

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"CRITICAL: Cannot find {input_file}. Did the download phase complete successfully?")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 1. Load all messages into a dictionary keyed by their unique message_id
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading messages"):
            row = json.loads(line)
            messages[row["message_id"]] = row

    print(f"Loaded {len(messages):,} total nodes. Untangling conversation trees...")

    # 2. Find all "leaf" nodes (messages that no other message claims as a parent)
    parent_ids = {msg["parent_id"] for msg in messages.values() if msg["parent_id"] is not None}
    leaves = [msg for msg in messages.values() if msg["message_id"] not in parent_ids]

    formatted_threads = []

    # 3. Trace backward from every leaf to the root to build a linear thread
    for leaf in tqdm(leaves, desc="Untangling trees"):
        # OASST has many languages. We only want to process English threads.
        if leaf.get("lang") != "en":
            continue

        thread = []
        current_msg = leaf

        # Trace back using the parent_id until we hit the root (where parent_id is None)
        while current_msg is not None:
            # Map OpenAssistant's roles to our standard roles
            role = "user" if current_msg["role"] == "prompter" else "assistant"
            
            # Insert at the beginning of the list since we are tracing backwards
            thread.insert(0, {
                "role": role,
                "content": current_msg["text"]
            })
            
            parent_id = current_msg.get("parent_id")
            current_msg = messages.get(parent_id) if parent_id else None

        # 4. Quality Control: Only keep valid threads
        # We ensure it starts with a user prompt and contains at least one response
        if len(thread) >= 2 and thread[0]["role"] == "user":
            thread.insert(0, {"role": "system", "content": persona})  # Add the system persona at the start
            formatted_threads.append({"messages": thread})

    print(f"Successfully reconstructed {len(formatted_threads):,} English conversation threads!")
    print(f"Saving formatted data to {output_file}...")

    # 5. Save in the standard format
    with open(output_file, 'w', encoding='utf-8') as f:
        for thread in tqdm(formatted_threads, desc="Saving threads"):
            f.write(json.dumps(thread, ensure_ascii=False) + '\n')

    print("OASST formatting complete!")

if __name__ == "__main__":
    format_oasst_data(input_file="data/raw/oasst_raw.jsonl", output_file="data/processed/oasst_formatted.jsonl")
