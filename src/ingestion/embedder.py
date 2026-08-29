import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "BAAI/bge-base-en-v1.5"


def load_model(model_id=MODEL_ID):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model, device


def load_chunks(chunks_path):
    records = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def embed_texts(texts, tokenizer, model, device, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            model_output = model(**encoded).last_hidden_state.mean(dim=1)
            embeddings = torch.nn.functional.normalize(model_output, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def save_embeddings(ids, embeddings, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(out_dir, "ids.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f)


if __name__ == "__main__":
    processed_dir = "/Users/laflame/project_aug/AI-Incident-Response-Copilot/data/processed"
    chunks_path = os.path.join(processed_dir, "chunks.jsonl")

    records = load_chunks(chunks_path)
    tokenizer, model, device = load_model()
    embeddings = embed_texts([r["text"] for r in records], tokenizer, model, device)

    save_embeddings([r["id"] for r in records], embeddings, processed_dir)
    print(f"Embedded {len(records)} chunks -> {embeddings.shape} saved to {processed_dir}")
