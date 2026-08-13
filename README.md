# GraphRAG

A GraphRAG pipeline applied to Gene Ontology (GO) term relationships: extract a knowledge graph, partition it into communities with the Leiden algorithm, summarize each community with an LLM, and validate the summaries with an LLM-as-judge scored against semantic similarity (Lin score).

Judged across 259 GO-term communities: **71.4% scored coherent**, with outlier-term cleaning raising the average judge score from 7.15 → 7.31 (see `llm_judge_se2_results_level2.json`).

**Live:** [graphrag-viewer.vercel.app](https://graphrag-viewer.vercel.app) — a filterable/searchable viewer over the judge output. Also mirrored via GitHub Pages at [devanshig16.github.io/GraphRAG](https://devanshig16.github.io/GraphRAG/). To run it locally: `python3 -m http.server` from this directory, then open `index.html` (it fetches the JSON directly, so it won't load over a plain `file://` URL).

## Pipeline

1. **Extraction** (`extract_graph.py`, `extract_directed_graph.py`) — builds a graph from GO term relationships, reduced to its giant connected component (`giant_component*.pkl`).
2. **Community detection** (`community_detection.py`, `se2_community.py`, `community_Lin.py`) — partitions the graph via Leiden clustering (`leidenalg`/`igraph`) and an alternative `speakeasy2`-based method, at multiple hierarchy levels (`communities*/`).
3. **Reassignment** (`community_reassign.py`) — flags and reassigns outlier terms within a community using Lin semantic similarity (`goatools`) against the GO DAG.
4. **Summarization** (`community_summarization.py`) — generates an LLM summary + theme per community, with and without outlier cleaning (`summaries/`, `summaries_comparison/`).
5. **Validation** (`community_validator.py`, `community_validator_depth.py`, `summary_comparison.py`) — LLM-as-judge scores each community summary for coherence before/after outlier removal (`llm_judge_*.json`).

## Tech stack

Python, NetworkX, python-igraph, leidenalg, speakeasy2, goatools (GO DAG + Lin semantic similarity), an LLM for summarization/judging.

## Setup

```bash
pip install -r requirements.txt
```

Requires a local GO DAG file (`go-basic.obo`, from the [Gene Ontology](https://geneontology.org/docs/download-ontology/) site) and, depending on which script you're running, either an OpenAI-compatible API key or a local Ollama install for the LLM summarization/judging steps.
