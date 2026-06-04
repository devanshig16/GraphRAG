# GraphRAG

A minimal, from-scratch implementation of **Graph RAG** (Retrieval-Augmented Generation with a Knowledge Graph) in ~120 lines of Python.

---

## What is GraphRAG?

Standard RAG retrieves the *k* most similar text chunks to your question and feeds them to an LLM. It works well for simple lookups but misses relationships between concepts spread across a document.

**GraphRAG** solves this by first converting the document into a **knowledge graph** — nodes are entities (people, places, concepts) and edges are the relationships between them. The graph is then partitioned into *communities* (clusters of closely related nodes), each summarised by an LLM. At query time the LLM answers using those community summaries, giving it a bird's-eye view of the whole document rather than a narrow slice.

---

## How it works — step by step

```
Raw text
   │
   ▼
1. Chunk          Split into overlapping windows of ~600 words
   │
   ▼
2. Extract        LLM pulls out entities + (entity1, relation, entity2) triples
   │
   ▼
3. Build graph    NetworkX graph: nodes = entities, edges = relationships
   │
   ▼
4. Communities    Greedy modularity algorithm groups tightly-connected nodes
   │
   ▼
5. Summarise      LLM writes a 2-3 sentence summary per community
   │
   ▼
6. Query          LLM reads all summaries and answers your question
```

### Why communities?
A document about, say, Marie Curie will produce one cluster around *Physics / radioactivity / Nobel Prize* and another around *World War I / X-ray units / hospitals*. A question about the war effort is answered from the second summary, not a random chunk — giving more coherent, complete answers.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your OpenAI key
cp .env.example .env
# then edit .env and paste your key
```

---

## Usage

```python
from graphrag import GraphRAG

rag = GraphRAG()

# Index any plain text
rag.index(open("my_document.txt").read())

# Ask questions
print(rag.query("What are the main themes?"))
print(rag.query("Who are the key people mentioned?"))
```

Run the built-in demo:

```bash
python graphrag.py
```

---

## File overview

| File | Purpose |
|------|---------|
| `graphrag.py` | Complete implementation — chunking, extraction, graph, communities, querying |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for your OpenAI API key |

---

## Dependencies

| Library | Used for |
|---------|---------|
| `openai` | Entity extraction, community summarisation, question answering |
| `networkx` | Knowledge graph construction and community detection |
| `python-dotenv` | Loading the API key from `.env` |

---

## Limitations

- Uses `gpt-4o-mini` for all LLM calls — indexing a long document will make several API calls (one per chunk + one per community).
- Community detection uses greedy modularity, which works well on mid-sized graphs but is not deterministic on large ones.
- No vector search — retrieval is purely graph-structural. For hybrid retrieval, combine with a vector store.
