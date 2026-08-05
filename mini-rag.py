from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass

import faiss
import pandas as pd
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------- config
DATA_PATH = os.getenv("DATA_PATH", "data/wiki_movie_plots_deduped.csv")
N_MOVIES = int(os.getenv("N_MOVIES", "300"))
MIN_YEAR = int(os.getenv("MIN_YEAR", "1990"))
CHUNK_WORDS = 300
CHUNK_OVERLAP = 50
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
TOP_K = int(os.getenv("TOP_K", "4"))
SNIPPET_CHARS = 400


# ---------------------------------------------------------------- 1. load & preprocess
@dataclass
class Chunk:
    movie_id: int
    title: str
    year: int
    chunk_id: int
    text: str


def load_movies(path: str = DATA_PATH, n: int = N_MOVIES) -> pd.DataFrame:
    """Load the Kaggle CSV and cut it down to a small, clean, demo-friendly subset."""
    if not os.path.exists(path):
        sys.exit(
            f"Dataset not found at {path}.\n"
            "Download 'wiki_movie_plots_deduped.csv' from Kaggle "
            "(jrobischon/wikipedia-movie-plots) and put it in ./data/"
        )

    df = pd.read_csv(path)
    df = df[["Release Year", "Title", "Origin/Ethnicity", "Plot"]].dropna()


    df = df[(df["Origin/Ethnicity"] == "American") & (df["Release Year"] >= MIN_YEAR)]
    df = df[df["Plot"].str.split().str.len() >= 150]
    df = df.drop_duplicates(subset="Title")

    df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    print(f"[1/4] Loaded {len(df)} movies from {path}")
    return df


# ---------------------------------------------------------------- 2. chunking
def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fixed-size sliding window over words. Simple, deterministic, no extra deps."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]

    step = size - overlap
    pieces = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if len(window) < 40 and pieces:
            break
        pieces.append(" ".join(window))
        if start + size >= len(words):
            break
    return pieces


def build_chunks(df: pd.DataFrame) -> list[Chunk]:
    chunks: list[Chunk] = []
    for movie_id, row in df.iterrows():
        for chunk_id, piece in enumerate(chunk_words(row["Plot"])):
            chunks.append(
                Chunk(
                    movie_id=int(movie_id),
                    title=row["Title"],
                    year=int(row["Release Year"]),
                    chunk_id=chunk_id,
                    text=piece,
                )
            )
    print(f"[2/4] Built {len(chunks)} chunks (~{CHUNK_WORDS} words, {CHUNK_OVERLAP} overlap)")
    return chunks


# ---------------------------------------------------------------- 3. embed & index
def build_index(chunks: list[Chunk], embedder: SentenceTransformer) -> faiss.Index:
    """Embed every chunk and load it into an in-memory FAISS index.

    Vectors are L2-normalised, so inner product == cosine similarity.
    The title+year prefix gives the embedding a strong 'which movie is this' signal.
    """
    texts = [f"{c.title} ({c.year}). {c.text}" for c in chunks]
    vectors = embedder.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    print(f"[3/4] Indexed {index.ntotal} vectors of dim {vectors.shape[1]}")
    return index


# ---------------------------------------------------------------- 4. retrieve
def retrieve(
    query: str,
    embedder: SentenceTransformer,
    index: faiss.Index,
    chunks: list[Chunk],
    k: int = TOP_K,
) -> list[tuple[Chunk, float]]:
    q = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, ids = index.search(q, k)
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


# ---------------------------------------------------------------- 5. generate
class LLMAnswer(BaseModel):
    """Schema the LLM is forced to fill. Contexts are added by us, not the model."""

    answer: str = Field(description="Natural-language answer, grounded only in the context.")
    reasoning: str = Field(description="2-3 sentences on how the answer was formed.")


SYSTEM_INSTRUCTION = """You answer questions about movies using ONLY the plot excerpts you are given.

Rules:
- Ground every claim in the numbered excerpts. Never use outside knowledge.
- Name the film(s) you relied on, with the year, in the answer.
- If the excerpts do not contain the answer, say so plainly instead of guessing.
- In `reasoning`, describe how you moved from the question to the excerpts to the answer,
  and mention which excerpt numbers you used."""


def build_prompt(query: str, hits: list[tuple[Chunk, float]]) -> str:
    context_block = "\n\n".join(
        f"[{i + 1}] {c.title} ({c.year}) — similarity {score:.3f}\n{c.text}"
        for i, (c, score) in enumerate(hits)
    )
    return (
        f"Retrieved plot excerpts:\n\n{context_block}\n\n"
        f"---\nQuestion: {query}"
    )


def generate(query: str, hits: list[tuple[Chunk, float]], client: genai.Client) -> LLMAnswer:
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=build_prompt(query, hits),
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "response_mime_type": "application/json",
            "response_schema": LLMAnswer,
            "temperature": 0.2,
        },
    )

    logger.debug("Raw LLM output: %s", response.text)

    return response.parsed


# ---------------------------------------------------------------- 6. structured output
def answer_query(query, embedder, index, chunks, client) -> dict:
    hits = retrieve(query, embedder, index, chunks)
    llm = generate(query, hits, client)



    return {
        "answer": llm.answer,
        "contexts": [
            f"{c.title} ({c.year}) ... {c.text[:SNIPPET_CHARS].rstrip()} ..."
            for c, _ in hits
        ],
        "reasoning": llm.reasoning
    }


# ---------------------------------------------------------------- CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Mini RAG over movie plots")
    parser.add_argument("--query", "-q", help="Run one query and exit")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GOOGLE_API_KEY in .env")
    client = genai.Client(api_key=api_key)

    df = load_movies()
    chunks = build_chunks(df)
    embedder = SentenceTransformer(EMBED_MODEL)
    index = build_index(chunks, embedder)
    print(f"[4/4] Ready. Using {LLM_MODEL}.\n")

    if args.query:
        result = answer_query(args.query, embedder, index, chunks, client)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    while True:
        try:
            q = input("Ask about a movie plot (Ctrl-C to quit)> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q:
            continue
        result = answer_query(q, embedder, index, chunks, client)
        print(json.dumps(result, indent=2, ensure_ascii=False), "\n")


if __name__ == "__main__":
    main()
