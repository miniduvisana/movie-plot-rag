# Mini RAG — Wikipedia Movie Plots

A minimal Retrieval-Augmented Generation pipeline that answers questions about movie
plots and returns structured JSON.

```
CSV subset → 300-word chunks → MiniLM embeddings → FAISS (in-memory)
           → top-k retrieval → Gemini 2.5 Flash → {answer, contexts, reasoning}
```

## Stack

| Stage | Choice                     | Why |
|---|----------------------------|---|
| Embeddings | `all-MiniLM-L6-v2` (local) | Free, no API key, 384-dim, ~20 MB, fast on CPU |
| Vector store | FAISS `IndexFlatIP`        | In-memory, exact search, cosine via normalised vectors |
| LLM | Gemini 3.6 Flash           | Free tier, no credit card, native JSON schema output |

Nothing here costs money.

## Setup

**1. Clone and install**

```bash
git clone https://github.com/<you>/mini-rag-movies.git
cd mini-rag-movies
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10–3.12 recommended (`faiss-cpu` wheels are most reliable there).

**2. Get the dataset**

Download `wiki_movie_plots_deduped.csv` from
[Kaggle: jrobischon/wikipedia-movie-plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
and place it at `data/wiki_movie_plots_deduped.csv`.

**3. Get a free API key**

Create one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env      # paste your key into GOOGLE_API_KEY
```

## Run

One-shot:

```bash
python mini_rag.py --query "Which movie has an artificial intelligence that turns against the crew?"
```

Interactive:

```bash
python mini_rag.py
```

First run downloads the embedding model (~90 MB) and takes ~30s to index.

## Output shape

```json
{
  "query": "Which movie features a hacker who discovers reality is a simulation?",
  "answer": "The Matrix (1999) follows Neo, a hacker who learns that the world he lives in is a simulated reality run by machines.",
  "contexts": [
    "The Matrix (1999) ... Thomas Anderson, a computer programmer who moonlights as the hacker Neo ... ..."
  ],
  "reasoning": "The question asked about a hacker and a simulated reality. Excerpt [1] describes Neo discovering the Matrix, so I answered from that excerpt."
}
```

`contexts` are the literal retrieved chunks — the model never writes them, so they can't
be hallucinated. `retrieval_debug` exposes the similarity scores so retrieval quality is
inspectable independently of generation.

## Configuration

Set in `.env` or as environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `N_MOVIES` | 300 | Size of the subset |
| `TOP_K` | 4 | Chunks retrieved per query |
| `LLM_MODEL` | `gemini-2.5-flash` | Use `gemini-2.5-flash-lite` for a higher daily quota |
| `MIN_YEAR` | 1990 | Earliest release year kept |

## Swapping the LLM

The only Gemini-specific code is `generate()` in `mini_rag.py`. To use Groq's free tier,
a local Ollama model, or OpenAI, replace that one function — retrieval is untouched.

## Known limitations

- Fixed-size word chunks can split a scene across boundaries; the 50-word overlap softens
  this but does not eliminate it.
- Dense retrieval alone struggles with exact proper nouns. A BM25 + vector hybrid would be
  the first upgrade.
- The index is rebuilt on every start. Fine at 300 movies; persist to disk beyond that.