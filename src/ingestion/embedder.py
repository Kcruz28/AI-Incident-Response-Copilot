import json
import os
import uuid

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient, models

MODEL_ID = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "incident_chunks"


def start_qdrant_client(host="localhost", port=6333):
    return QdrantClient(host=host, port=port)


def stop_qdrant_client(client):
    client.close()


def start_collection(client, collection_name, vector_size):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
    )


def add_embeddings(client, collection_name, records, embeddings):
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, record["id"])),
            vector=embedding.tolist(),
            payload={
                "chunk_id": record["id"],
                "source": record["source"],
                "text": record["text"],
            },
        )
        for record, embedding in zip(records, embeddings)
    ]
    client.upsert(collection_name=collection_name, points=points)
    return points


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


if __name__ == "__main__":
    processed_dir = "/Users/laflame/project_aug/AI-Incident-Response-Copilot/data/processed"
    chunks_path = os.path.join(processed_dir, "chunks.jsonl")

    records = load_chunks(chunks_path)
    tokenizer, model, device = load_model()
    embeddings = embed_texts([r["text"] for r in records], tokenizer, model, device)

    client = start_qdrant_client()
    start_collection(client, COLLECTION_NAME, vector_size=embeddings.shape[1])
    add_embeddings(client, COLLECTION_NAME, records, embeddings)
    stop_qdrant_client(client)

    print(f"Embedded {len(records)} chunks -> {embeddings.shape} and upserted into Qdrant collection '{COLLECTION_NAME}'")
