import json
import os

def raw_data(dir_path):
    files_names = sorted(os.listdir(dir_path))
    files_paths = []
    for file_name in files_names:
        file_path = os.path.join(dir_path, file_name)
        if file_name.endswith(".txt") and os.path.isfile(file_path):
            files_paths.append(file_path)
    return files_paths

def reading(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=150, overlap=30):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
    return chunks

def chunk_documents(dir_path, chunk_size=150, overlap=30):
    records = []
    for file_path in raw_data(dir_path):
        source = os.path.basename(file_path)
        text = reading(file_path)
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            records.append({
                "id": f"{source}::{i}",
                "source": source,
                "text": chunk,
            })
    return records

def save_chunks(records, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    data_dir = "/Users/laflame/project_aug/AI-Incident-Response-Copilot/data/raw"
    out_path = "/Users/laflame/project_aug/AI-Incident-Response-Copilot/data/processed/chunks.jsonl"
    records = chunk_documents(data_dir)
    save_chunks(records, out_path)
    print(f"Wrote {len(records)} chunks to {out_path}")