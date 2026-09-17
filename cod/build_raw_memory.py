from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import DEFAULT_CONFIG_PATH, load_config
from .schema import RawMemoryEntry
from .store import create_embedder


def parse_timestamp(value: str) -> str:
    parsed = datetime.strptime(value.strip("()"), "%I:%M %p on %d %B, %Y")
    return parsed.isoformat(sep=" ")


def raw_entries_from_sample(sample: dict, embeddings: list[list[float]]) -> list[RawMemoryEntry]:
    conversation_id = sample["sample_id"]
    conversation = sample["conversation"]
    entries: list[RawMemoryEntry] = []
    embedding_index = 0
    session_numbers = sorted(
        int(key.split("_")[1])
        for key in conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    )
    for session_number in session_numbers:
        session_key = f"session_{session_number}"
        timestamp = parse_timestamp(conversation[f"{session_key}_date_time"])
        for turn_index, turn in enumerate(conversation.get(session_key, [])):
            entries.append(
                RawMemoryEntry(
                    id=f"{conversation_id}/{session_key}/turn_{turn_index}",
                    conversation_id=conversation_id,
                    speaker=str(turn["speaker"]),
                    text=str(turn["text"]),
                    timestamp=timestamp,
                    embedding=embeddings[embedding_index],
                )
            )
            embedding_index += 1
    return entries


def sample_texts(sample: dict) -> list[str]:
    conversation = sample["conversation"]
    session_numbers = sorted(
        int(key.split("_")[1])
        for key in conversation
        if key.startswith("session_") and not key.endswith("_date_time")
    )
    return [
        str(turn["text"])
        for number in session_numbers
        for turn in conversation.get(f"session_{number}", [])
    ]


def embed_texts(embedder, texts: list[str], batch_size: int) -> list[list[float]]:
    if hasattr(embedder, "model"):
        vectors = embedder.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        embedder.total_calls += len(texts)
        return vectors.tolist()
    return [embedder.embed(text) for text in texts]


def build_raw_store(
    data_path: str,
    output_dir: str,
    embed_model: str,
    device: str,
    embedding_dims: int,
    batch_size: int = 64,
    overwrite: bool = False,
) -> dict:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    with Path(data_path).open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    embedder = create_embedder(embed_model, device)
    client = QdrantClient(path=str(destination))
    counts: dict[str, int] = {}
    try:
        for sample in dataset:
            conversation_id = sample["sample_id"]
            exists = client.collection_exists(conversation_id)
            if exists and not overwrite:
                counts[conversation_id] = client.count(conversation_id).count
                continue
            if exists:
                client.delete_collection(conversation_id)
            client.create_collection(
                collection_name=conversation_id,
                vectors_config=VectorParams(
                    size=embedding_dims,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            )
            texts = sample_texts(sample)
            embeddings = embed_texts(embedder, texts, batch_size)
            entries = raw_entries_from_sample(sample, embeddings)
            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, entry.id)),
                    vector=entry.embedding,
                    payload=entry.to_dict(include_embedding=False),
                )
                for entry in entries
            ]
            for start in range(0, len(points), 128):
                client.upsert(conversation_id, points[start : start + 128], wait=True)
            counts[conversation_id] = len(entries)
    finally:
        client.close()

    manifest = {
        "representation": "raw",
        "schema": ["id", "conversation_id", "speaker", "text", "timestamp", "embedding"],
        "embed_model": embed_model,
        "embedding_dims": embedding_dims,
        "data_path": data_path,
        "collections": counts,
        "total_entries": sum(counts.values()),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build minimally transformed LoCoMo memory")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = build_raw_store(
        data_path=config["data_path"],
        output_dir=args.output_dir or config["raw_store_dir"],
        embed_model=config["embed_model"],
        device=config.get("device", "cuda"),
        embedding_dims=int(config.get("embedding_dims", 384)),
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
