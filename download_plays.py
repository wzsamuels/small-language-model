import os
import urllib.request
import time

input_file = 'play_ids.txt'
output_dir = 'gutenberg_plays'

os.makedirs(output_dir, exist_ok=True)

print("Loading IDs...")
with open(input_file, 'r') as f:
    play_ids = [line.strip() for line in f if line.strip()]

print(f"Starting download for {len(play_ids)} plays...")

for pg_id in play_ids:
    url = f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt"
    output_path = os.path.join(output_dir, f"{pg_id}.txt")

    if os.path.exists(output_path):
        print(f"Skipping {pg_id} - already exists.")
        continue

    print(f"Downloading PG ID {pg_id}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        time.sleep(1)
    except Exception as e:
        print(f"Failed to download {pg_id}: {e}")

print("Download complete!")
