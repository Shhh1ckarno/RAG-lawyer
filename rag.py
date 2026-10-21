import chromadb
from openai import OpenAI

from config import (
    BASE_URL,
    BATCH_SIZE,
    COLLECTION_NAME,
    DB_PATH,
    EMBED_MODEL,
    REWRITTEN_PATH,
)
from parts import (
    NOTE_PART,
    all_parts,
    keeps_points,
    load_rewritten,
    part_label,
    plain_disposition,
    split_points,
)


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return client, collection


def get_embedding_client() -> OpenAI:
    return OpenAI(
        base_url=BASE_URL,
        api_key="lm_studio",
        timeout=60,
    )


def embed(texts: list[str]) -> list[list[float]]:
    response = get_embedding_client().embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [
        item.embedding
        for item in sorted(response.data, key=lambda item: item.index)
    ]


def part_records(part: dict) -> list[dict]:
    if not split_points(part["disposition"])[1]:
        return [{**part, "point": "", "lead": part["rewritten"], "point_text": ""}]
    text = (
        part["rewritten"] if keeps_points(part, part["rewritten"])
        else plain_disposition(part)
    )
    lead, points = split_points(text)
    return [
        {
            **part,
            "id": f"{part['id']}_{letter}",
            "point": letter,
            "lead": lead,
            "point_text": body,
        }
        for letter, body in points
    ]


def record_text(record: dict) -> str:
    if not record["point_text"]:
        return record["lead"]
    return f"{record['lead'].rstrip(' :,-–—')} {record['point_text']}."


def part_document(record: dict) -> str:
    return (
        f"Статья {record['article_number']}. {record['title']}\n"
        f"{part_label(record).capitalize()}\n"
        f"{record_text(record)}"
    )


def index_document() -> None:
    actual_ids = {part["id"] for part in all_parts()}
    parts = [
        record
        for part in load_rewritten().values()
        if part["id"] in actual_ids
        for record in part_records(part)
    ]
    if not parts:
        raise FileNotFoundError(
            f"Нет данных в {REWRITTEN_PATH.name}. "
            "Сначала выполни: python parts.py"
        )

    client = chromadb.PersistentClient(path=str(DB_PATH))

    existing_names = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }
    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Записей (частей и пунктов) к индексации: {len(parts)}")

    for start in range(0, len(parts), BATCH_SIZE):
        batch = parts[start:start + BATCH_SIZE]
        documents = [part_document(part) for part in batch]

        collection.upsert(
            ids=[part["id"] for part in batch],
            documents=documents,
            embeddings=embed(documents),
            metadatas=[
                {
                    "article_number": part["article_number"],
                    "title": part["title"],
                    "part": part["part"],
                    "point": part["point"],
                    "lead": part["lead"],
                    "point_text": part["point_text"],
                    "original": part["text"],
                    "punishment": part["punishment"],
                    "order": start + index,
                }
                for index, part in enumerate(batch)
            ],
        )

        print(
            f"Сохранено: "
            f"{min(start + BATCH_SIZE, len(parts))}/{len(parts)}"
        )

    print("Индексация завершена.")
    print(f"Всего записей: {collection.count()}")


def search_parts(
    queries: list[str],
    n_results: int = 10,
    include_notes: bool = False,
) -> list[dict]:
    _, collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "База пуста. Выполни: python parts.py, затем python rag.py"
        )

    queries = [query for query in queries if query.strip()]
    where = None if include_notes else {"part": {"$ne": NOTE_PART}}

    results = collection.query(
        query_embeddings=embed(queries),
        n_results=min(n_results, collection.count()),
        where=where,
        include=["metadatas", "distances"],
    )

    best: dict[str, dict] = {}
    for ids, metadatas, distances in zip(
        results["ids"], results["metadatas"], results["distances"]
    ):
        for part_id, metadata, distance in zip(ids, metadatas, distances):
            if part_id not in best or distance < best[part_id]["distance"]:
                best[part_id] = {
                    "id": part_id,
                    **metadata,
                    "distance": distance,
                }

    return sorted(best.values(), key=lambda item: item["distance"])


def get_article_parts(article_number: str) -> list[dict]:
    _, collection = get_collection()
    result = collection.get(
        where={"article_number": article_number},
        include=["metadatas"],
    )
    parts = result["metadatas"]
    return sorted(
        parts,
        key=lambda part: (
            part["part"] == NOTE_PART,
            int(part["part"]) if part["part"].isdigit() else 0,
            part["order"],
        ),
    )


if __name__ == "__main__":
    index_document()
