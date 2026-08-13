import json
import pickle
import requests
import os
from collections import defaultdict

print("="*60)
print("GRAPHRAG OUTLIER COMPARISON ANALYSIS")
print("="*60)

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

# Limit to 10 target communities for this test
TARGET_LIMIT = 10

# Input files
COMMUNITIES_FILE = "communities_4/level_2_communities.json"
OUTLIERS_FILE = "llm_judge_results.json"  # Ensure this matches your outlier filename
GRAPH_FILE = "giant_component_directed.pkl"

# Output directories
OUTPUT_DIR_WITH = "summaries_comparison/with_outliers"
OUTPUT_DIR_WITHOUT = "summaries_comparison/without_outliers"
os.makedirs(OUTPUT_DIR_WITH, exist_ok=True)
os.makedirs(OUTPUT_DIR_WITHOUT, exist_ok=True)

# ============================================================
# PROMPT TEMPLATE (Same as your original)
# ============================================================

PROMPT_TEMPLATE = """You are a Gene Ontology (GO) expert with deep knowledge of biological systems.
You are analyzing a community of nodes and relationships extracted from a GO knowledge graph.

CRITICAL: You already know what every GO:XXXXXXX term means from your training data.
Treat GO IDs as meaningful biological concepts, NOT as opaque identifiers.
Always refer to GO terms by their biological meaning, not just their ID.

---

TASK: Generate a structured community report describing the biological mechanisms this community represents.

---

RULES:
1. Use ONLY the provided entities and relationships as your evidence base
2. Every finding MUST be derived from a specific relationship in the relationships table — not just node presence
3. Chemical entities (non-GO nodes) must be explicitly addressed — do not ignore them

---

STRUCTURE YOUR OUTPUT AS:

Return output as valid JSON:
{{
  "title": "<short specific title representing key entities>",
  "summary": "<executive summary of community structure and entity relationships>",
  "findings": [
    {{
      "summary": "<short insight>",
      "explanation": "<detailed explanation with 2-3 paragraphs>"
    }}
  ],
  "rating": <float between 0-10 representing biological importance>,
  "rating_explanation": "<single sentence explaining the rating>"
}}


Provide 5-10 findings. Return ONLY valid JSON, no markdown, no extra text.

---Input---

Entities
id, entity
{entities_table}

Relationships
id, source, target, relationship
{relationships_table}

---Output---
"""

# ============================================================
# LOAD DATA & NORMALIZE
# ============================================================

print(f"\nLoading communities...")
with open(COMMUNITIES_FILE) as f:
    communities = json.load(f)

print(f"Loading outlier information...")
with open(OUTLIERS_FILE) as f:
    outlier_list = json.load(f)

print(f"Loading directed graph...")
with open(GRAPH_FILE, 'rb') as f:
    G = pickle.load(f)

# Helper to normalize GO terms for matching
def norm(term): return term.strip().lower()

# Map community IDs to their identified outliers
outlier_map = {}
for entry in outlier_list:
    cid = str(entry['community_id'])
    # Store outliers in a set for fast lookup
    outlier_map[cid] = {norm(o) for o in entry.get('outliers', [])}

# ============================================================
# SUMMARIZATION FUNCTION
# ============================================================

def summarize_community(comm_id, nodes):
    """
    Generate GraphRAG summary for a specific node set.
    """
    # Build entities table
    entities_lines = [f"{i}, {node}" for i, node in enumerate(nodes)]
    entities_table = "\n".join(entities_lines)
    
    # Build relationships table (Filtering for edges where BOTH nodes exist in current set)
    node_set = set(nodes)
    relationships_lines = []
    rel_id = 0
    for u, v, data in G.edges(data=True):
        if u in node_set and v in node_set:
            rel = data.get('relationship', 'related')
            relationships_lines.append(f"{rel_id}, {u}, {v}, {rel}")
            rel_id += 1
    
    relationships_table = "\n".join(relationships_lines) if relationships_lines else "(no relationships within community)"
    
    prompt = PROMPT_TEMPLATE.format(
        entities_table=entities_table,
        relationships_table=relationships_table
    )
    
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 3000}
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        response_text = response.json().get('response', '').strip()
        
        # Clean markdown code blocks
        if response_text.startswith('```'):
            response_text = "\n".join(response_text.split('\n')[1:-1])
            
        return json.loads(response_text)
    except Exception as e:
        print(f" Error: {e}")
        return None

# ============================================================
# PROCESS COMPARISON
# ============================================================

# 1. Identify communities that actually have outliers
target_communities = [cid for cid, outs in outlier_map.items() if len(outs) > 0 and cid in communities]
target_communities = target_communities[:TARGET_LIMIT]

print(f"\nFound {len(target_communities)} communities with outliers. Starting comparison...\n")

for i, cid in enumerate(target_communities):
    original_nodes = communities[cid]
    outliers_to_remove = outlier_map[cid]
    
    # Create Cleaned Node Set (Non-outliers)
    cleaned_nodes = [n for n in original_nodes if norm(n) not in outliers_to_remove]
    
    print(f"[{i+1}/{TARGET_LIMIT}] Community {cid}:")
    print(f"  - Original: {len(original_nodes)} nodes")
    print(f"  - Cleaned:  {len(cleaned_nodes)} nodes (Removed: {len(outliers_to_remove)})")

    # Save filename
    safe_name = cid.replace('.', '_') + ".json"

    # --- PART 1: WITH OUTLIERS ---
    if not os.path.exists(os.path.join(OUTPUT_DIR_WITH, safe_name)):
        print("    Generating 'With Outliers'...", end=" ", flush=True)
        summary_with = summarize_community(cid, original_nodes)
        if summary_with:
            with open(os.path.join(OUTPUT_DIR_WITH, safe_name), 'w') as f:
                json.dump(summary_with, f, indent=2)
            print("✓")
    else:
        print("    'With Outliers' exists. Skipping.")

    # --- PART 2: WITHOUT OUTLIERS ---
    if not os.path.exists(os.path.join(OUTPUT_DIR_WITHOUT, safe_name)):
        print("    Generating 'Without Outliers'...", end=" ", flush=True)
        summary_without = summarize_community(cid, cleaned_nodes)
        if summary_without:
            with open(os.path.join(OUTPUT_DIR_WITHOUT, safe_name), 'w') as f:
                json.dump(summary_without, f, indent=2)
            print("✓")
    else:
        print("    'Without Outliers' exists. Skipping.")

print(f"\n{'='*60}")
print("COMPARISON COMPLETE")
print(f"Check results in: {OUTPUT_DIR_WITH} and {OUTPUT_DIR_WITHOUT}")
print(f"{'='*60}")