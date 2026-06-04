import os
import json
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── Step 1: Chunk text ────────────────────────────────────────────────────────

def chunk_text(text, chunk_size=600, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks


# ── Step 2: Extract entities + relationships from one chunk ───────────────────

def extract_graph_elements(chunk):
    prompt = f"""Extract entities and relationships from the text below.
Return a JSON object with exactly two keys:
- "entities": list of entity name strings (people, places, orgs, concepts)
- "relationships": list of [entity1, relationship_label, entity2] triples

Text:
{chunk}

Return only valid JSON."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, KeyError):
        return {"entities": [], "relationships": []}


# ── Step 3: Build the knowledge graph ────────────────────────────────────────

def build_knowledge_graph(text):
    graph = nx.Graph()
    for chunk in chunk_text(text):
        elements = extract_graph_elements(chunk)
        for entity in elements.get("entities", []):
            if entity and isinstance(entity, str):
                graph.add_node(entity.strip())
        for rel in elements.get("relationships", []):
            if isinstance(rel, list) and len(rel) == 3:
                src, label, dst = rel
                if src and dst:
                    graph.add_edge(src.strip(), dst.strip(), relation=label)
    return graph


# ── Step 4: Detect communities ────────────────────────────────────────────────

def detect_communities(graph):
    if len(graph.nodes) == 0:
        return {}
    raw = nx.algorithms.community.greedy_modularity_communities(graph)
    return {i: list(c) for i, c in enumerate(raw)}


# ── Step 5: Summarize each community ─────────────────────────────────────────

def summarize_community(nodes, graph):
    node_set = set(nodes)
    edges = [
        f"{u} {d.get('relation', 'relates to')} {v}"
        for u, v, d in graph.edges(data=True)
        if u in node_set and v in node_set
    ]
    context = f"Entities: {', '.join(nodes)}\nRelationships: {'; '.join(edges)}"

    prompt = f"""Given these entities and relationships from a knowledge graph, write a 2-3 sentence summary of what this group of concepts is about.

{context}

Summary:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ── Step 6: Answer a question using community summaries ──────────────────────

def query_graph(question, community_summaries):
    summaries_text = "\n\n".join(
        f"Community {i}:\n{s}" for i, s in community_summaries.items()
    )
    prompt = f"""You are answering a question using knowledge extracted from a document and organised into a knowledge graph.

Community summaries:
{summaries_text}

Question: {question}

Answer concisely and accurately based only on the information above."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ── Public API ────────────────────────────────────────────────────────────────

class GraphRAG:
    def __init__(self):
        self.graph = None
        self.community_summaries = {}

    def index(self, text):
        """Build the knowledge graph from raw text."""
        print("Building knowledge graph...")
        self.graph = build_knowledge_graph(text)
        print(f"  {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        print("Detecting communities...")
        communities = detect_communities(self.graph)
        print(f"  {len(communities)} communities found")

        print("Summarising communities...")
        self.community_summaries = {
            i: summarize_community(nodes, self.graph)
            for i, nodes in communities.items()
        }
        print("Done — ready to query.\n")

    def query(self, question):
        """Answer a question over the indexed text."""
        if not self.community_summaries:
            raise ValueError("Nothing indexed yet. Call index() first.")
        return query_graph(question, self.community_summaries)


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_text = """
    Marie Curie was a Polish-French physicist and chemist who conducted pioneering
    research on radioactivity. She was the first woman to win a Nobel Prize, the
    only person to win it twice, and the only person to win in two different sciences
    (Physics and Chemistry).

    Marie Curie discovered two elements: polonium, named after her homeland Poland,
    and radium. She was born in Warsaw in 1867 and later moved to Paris to study at
    the University of Paris. Her husband Pierre Curie was also a physicist and they
    collaborated closely. Pierre shared the 1903 Nobel Prize in Physics with Marie
    and Henri Becquerel.

    Marie Curie's work laid the foundation for nuclear physics. She founded the
    Curie Institutes in Paris and Warsaw. During World War I she developed mobile
    X-ray units to help battlefield surgeons.
    """

    rag = GraphRAG()
    rag.index(sample_text)

    for question in [
        "What did Marie Curie discover?",
        "What Nobel Prizes did Marie Curie win?",
        "Who was Pierre Curie?",
    ]:
        print(f"Q: {question}")
        print(f"A: {rag.query(question)}\n")
