import os
import re
import json
import random
from config import personas


def extract_speaker(line):
    """
    Tests a line against multiple known Gutenberg speaker formats.
    Returns a tuple: (Speaker Name, Dialogue on same line) or (None, None)
    """
    # 1. Pure ALL CAPS on its own line (e.g., "NAME")
    # Must be at least 2 characters to avoid matching words like "I" or "A"
    match = re.match(r'^([A-Z\s]{2,})$', line)
    if match: 
        return match.group(1).strip(), ""
        
    # 2. Contains Stage Directions (e.g., "Name (_Emotion_).")
    # Captures the name, ignores the parens, looks for optional punctuation/underscores
    match = re.match(r'^_?([A-Za-z\s]+)_?\s*\([^)]+\)_?[.:]_?\s*(.*)$', line)
    if match: 
        return match.group(1).strip(), match.group(2).strip()
        
    # 3. Standard with period/colon, optional underscores 
    # Matches: "_Name_.", "_Name._", "Name.", "_Name_:", "_Name:_"
    # We limit the name to 35 characters to prevent accidentally matching 
    # an entire standard sentence that happens to end in a period.
    match = re.match(r'^_?([A-Z][A-Za-z\s]{0,35})_?[.:]_?\s*(.*)$', line)
    if match: 
        return match.group(1).strip(), match.group(2).strip()
        
    return None, None

def process_plays(input_dir, output_jsonl, error_log):
    speaker_pattern = re.compile(r'^([A-Z\s]+)[:\.]\s*(.*)')

    success_count = 0
    fail_count = 0

    with open(output_jsonl, 'w', encoding='utf-8') as out_file, \
         open(error_log, 'w', encoding='utf-8') as error_log:

        for filename in os.listdir(input_dir):
            if not filename.endswith('.txt'):
                continue

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

            if match_count < 20:
                fail_count += 1
                error_log.write(f"--- FAILED: {filename} (Only {match_count} matches) ---\n")

                snippet_start = min(500, len(play_lines) // 2)
                snippet = "\n".join([l.strip() for l in play_lines[snippet_start:snippet_start+15] if l.strip()])
                
                error_log.write(f"SNIPPET FOR DEBUGGING:\n{snippet}\n\n")

            else:
                success_count += 1
                for i in range(len(conversation) - 1):
                    assigned_persona = random.choice(personas)
                    chat_entry = {
                        "messages" : [
                            {"role": "system", "content": assigned_persona},
                            {"role": "user", "content": conversation[i]["text"]},
                            {"role": "assistant", "content": conversation[i+1]["text"]}
                        ]
                    }
                    out_file.write(json.dumps(chat_entry) + '\n')

    print(f"Extraction Complete!")
    print(f"Successfully Formatted: {success_count} plays")
    print(f"Failed (Check Error Log): {fail_count} plays")

# Execute the pipeline
# Ensure 'gutenberg_plays' exists and contains your downloaded texts
process_plays('gutenberg_plays', 'training_data.jsonl', 'regex_failures.log')
