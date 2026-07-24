from datasets import load_dataset
# Login using e.g. `huggingface-cli login` to access this dataset

def format_books1_data(output_file, num_books = 5000):
    dataset = load_dataset("institutional/institutional-books-1.0", split="train", streaming=True)

    ds = next(iter(dataset))
    #print(dataset.column_names)
    #print(ds["text_by_page_gen"])

    with open(output_file, mode='w', encoding='utf-8') as file:
        for idx, book in enumerate(dataset):
            if idx >= num_books:
                break
            if book.get("language_src") != "eng":
                break
            raw_text = book.get("text_by_page_gen", "")
            #clean_text = " ".join(raw_text.split())
            for idx, text in enumerate(raw_text):
                print(f"{idx}: {text}")


if __name__ == "__main__":
    format_books1_data(output_file="data/raw/books1.jsonl", num_books=1)
