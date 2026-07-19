import os
import re
import json
import random
from tqdm import tqdm

persona = "You are an anarchist British punk rocker."

def clean_gutenberg_text(text):
    """Removes Gutenberg-specific formatting artifacts and stage directions."""
    # 1. Remove footnotes like [367] or [1]
    text = re.sub(r'\[\d+\]', '', text)
    # 2. Remove stage directions or anything else in brackets like [_Exeunt._]
    text = re.sub(r'\[.*?\]', '', text)
    # 3. Remove underscores used for italics
    text = text.replace('_', '')
    # 4. Clean up the whitespace (removes massive gaps left behind by deleted tags)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_speaker(line):
    """
    Tests a line against multiple known Gutenberg speaker formats.
    Returns a tuple: (Speaker Name, Dialogue on same line) or (None, None)
    """
    # 1. Pure ALL CAPS on its own line (e.g., "NAME")
    match = re.match(r'^([A-Z\s]{2,})$', line)
    if match: 
        return match.group(1).strip(), ""
        
    # 2. Contains Stage Directions (e.g., "Name (_Emotion_).")
    match = re.match(r'^_?([A-Za-z\s]+)_?\s*\([^)]+\)_?[.:]_?\s*(.*)$', line)
    if match: 
        return match.group(1).strip(), match.group(2).strip()
        
    # 3. Standard with period/colon, optional underscores 
    match = re.match(r'^_?([A-Z][A-Za-z\s]{0,35})_?[.:]_?\s*(.*)$', line)
    if match: 
        return match.group(1).strip(), match.group(2).strip()
        
    return None, None

def format_gutenberg_plays(input_dir, output_file, error_file = 'logs/gutenberg_errors.txt'):

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping Gutenberg play formatting.")
        return

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"CRITICAL: Cannot find {input_dir}. Did the download phase complete successfully?")

    success_count = 0
    fail_count = 0
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as out_file, \
         open(error_file, 'w', encoding='utf-8') as err_file:

        # Grab valid files
        play_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

        # Use tqdm for a beautiful terminal progress bar
        for filename in tqdm(play_files, desc="Parsing Plays"):
            filepath = os.path.join(input_dir, filename)

            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            start_idx, end_idx = 0, len(lines)
            for i, line in enumerate(lines):
                if "*** START OF THE PROJECT GUTENBERG" in line:
                    start_idx = i + 1
                elif "*** END OF THE PROJECT GUTENBERG" in line:
                    end_idx = i
                    break

            play_lines = lines[start_idx:end_idx]
            conversation = []
            current_speaker = None
            current_dialogue = []
            match_count = 0

            for line in play_lines:
                line = line.strip()
                if not line:
                    continue

                speaker_match, dialogue_match = extract_speaker(line)
                if speaker_match:
                    match_count += 1

                    if current_speaker and current_dialogue:
                        conversation.append({
                            "speaker": current_speaker,
                            "text": " ".join(current_dialogue)
                        })

                    current_speaker = speaker_match
                    current_dialogue = [dialogue_match] if dialogue_match else []
                elif current_speaker:
                    current_dialogue.append(line)

            # FIX: Catch the very last line of dialogue in the play before the loop ends!
            if current_speaker and current_dialogue:
                conversation.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_dialogue)
                })

            if match_count < 20:
                fail_count += 1
                err_file.write(f"--- FAILED: {filename} (Only {match_count} matches) ---\n")
                snippet_start = min(500, len(play_lines) // 2)
                snippet = "\n".join([l.strip() for l in play_lines[snippet_start:snippet_start+15] if l.strip()])
                err_file.write(f"SNIPPET FOR DEBUGGING:\n{snippet}\n\n")

            else:
                success_count += 1
                
                # FIX: Multi-turn chunking! Group dialogue into chunks of 8 turns
                chunk_size = 8
                for i in range(0, len(conversation), chunk_size):
                    chunk = conversation[i : i + chunk_size]
                    
                    # We need at least a prompt and a response to make a valid training example
                    if len(chunk) < 2: 
                        continue 
                        
                    messages = [{"role": "system", "content": persona}]
                    
                    # Alternate roles between 'user' and 'assistant' for the chunk
                    for j, turn in enumerate(chunk):
                        role = "user" if j % 2 == 0 else "assistant"
                        
                        # Apply our regex cleaner to destroy the Gutenberg formatting artifacts
                        clean_text = clean_gutenberg_text(turn["text"])
                        
                        if clean_text:
                            messages.append({"role": role, "content": clean_text})
                    
                    # Write the multi-turn thread to the file
                    if len(messages) > 1:
                        chat_entry = {"messages": messages}
                        out_file.write(json.dumps(chat_entry, ensure_ascii=False) + '\n')

    print(f"\nExtraction Complete!")
    print(f"Successfully Formatted: {success_count} plays")
    print(f"Failed (Check Error Log): {fail_count} plays")

if __name__ == "__main__":
    format_gutenberg_plays(input_dir="data/raw/gutenberg_plays", output_file="data/processed/gutenberg_formatted.jsonl", error_file="logs/gutenberg_errors.txt")