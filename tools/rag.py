import os
import re
from pathlib import Path

import chromadb
import ollama

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION = f"knowledge_{EMBED_MODEL.replace('-', '_').replace(':', '_')}"
# nomic-embed-text attend des préfixes ; bge-m3 n'en utilise pas
USE_PREFIX = EMBED_MODEL.startswith("nomic")


def embed(texts: list[str], kind: str) -> list[list[float]]:
    """kind : 'document' ou 'query'."""
    if USE_PREFIX:
        texts = [f"search_{kind}: {t}" for t in texts]
    return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def chunk_by_section(text: str) -> list[tuple[str, str]]:
    """Un passage par section '##', préfixé par le titre du document."""
    title_match = re.search(r"^# (.+)$", text, flags=re.M)
    title = title_match.group(1).strip() if title_match else ""
    chunks = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        heading, _, body = block.partition("\n")
        heading, body = heading.strip(), body.strip()
        if body:
            chunks.append((heading, f"{title} — {heading} : {body}"))
    return chunks


def build_index(knowledge_dir: str = "data/knowledge", db_path: str = "chroma_db") -> int:
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for f in sorted(Path(knowledge_dir).glob("*.md")):
        for i, (heading, chunk) in enumerate(chunk_by_section(f.read_text(encoding="utf-8"))):
            ids.append(f"{f.stem}#{i}")
            docs.append(chunk)
            metas.append({"source": f.stem, "section": heading})

    col.add(ids=ids, documents=docs, embeddings=embed(docs, "document"), metadatas=metas)
    return len(ids)


def search(
    query: str, k: int = 3, db_path: str = "chroma_db", sources: list[str] | None = None
) -> list[dict]:
    col = chromadb.PersistentClient(path=db_path).get_collection(COLLECTION)
    where = {"source": {"$in": sources}} if sources else None
    r = col.query(query_embeddings=embed([query], "query"), n_results=k, where=where)
    return [
        {"id": i, "source": m["source"], "section": m["section"], "text": d, "distance": round(dist, 3)}
        for i, d, m, dist in zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]
        )
    ]