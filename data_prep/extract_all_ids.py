import pandas as pd

csv_file_path = 'data/metadata.csv'
output_file_path = 'data/gutenberg_ids.txt'

print('Loading metadata...')

df = pd.read_csv(csv_file_path, low_memory=False)

print(f"Totat texts {len(df)}")

english_texts = df[df['language'] == "['en']"]
print(f"English texts: {len(df)}")

play_ids = english_texts['id'].tolist()

with open(output_file_path, 'w') as f:
    for text_id in play_ids:
        clean_id = str(text_id).replace('PG', '')
        f.write(f"{clean_id}\n")

print(f"Saved {len(play_ids)} to {output_file_path}")
