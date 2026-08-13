import json
import requests
import os
from collections import defaultdict
from goatools.base import get_godag

# ==============================
# CONFIG
# ==============================
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "qwen2.5:7b"
JUDGE_FILE  = "llm_judge_results.json"
EDGES_FILE  = "giant_component_edges.txt"
GO_OBO_FILE = "go-basic.obo"
OUTPUT_FILE = "communities_4/level_2_reassigned.json"
TOP_N       = 3

# ==============================
# Step 1: Load GO DAG
# ==============================
print("Loading GO DAG...")
godag = get_godag(GO_OBO_FILE, optional_attrs={"def", "synonym", "xref", "property_value"})
print(f"Loaded {len(godag)} GO terms")

# ==============================
# Step 2: Load edges — normalize to UPPERCASE
# ==============================
print("Loading edges...")
edge_map = defaultdict(set)
with open(EDGES_FILE) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            src = parts[0].upper()
            tgt = parts[1].upper()
            edge_map[src].add(tgt)
            edge_map[tgt].add(src)
print(f"Loaded {len(edge_map)} nodes in edge_map")

# ==============================
# Step 3: Load judge — build comm_nodes from cleaned_terms only
# ==============================
print("Loading judge results...")
with open(JUDGE_FILE) as f:
    judge_results = json.load(f)

comm_nodes  = {}  # comm_id -> cleaned_terms (uppercase)
comm_scores = {}  # comm_id -> final_score

for r in judge_results:
    if r.get("status") != "ok":
        continue
    comm_nodes[r["community_id"]]  = [t.upper() for t in r["cleaned_terms"]]
    comm_scores[r["community_id"]] = r["final_score"]

# ==============================
# Step 4: Build node -> community from cleaned_terms
# ==============================
node_to_comm = {}
for comm_id, terms in comm_nodes.items():
    for term in terms:
        node_to_comm[term] = comm_id

# ==============================
# Step 5: Collect outliers — uppercase
# ==============================
outliers_to_reassign = []
for r in judge_results:
    if r.get("status") != "ok" or not r.get("outliers"):
        continue
    for term in r["outliers"]:
        outliers_to_reassign.append((term.upper(), r["community_id"]))

print(f"Total outliers to reassign: {len(outliers_to_reassign)}")

# ==============================
# Step 6: Exact same helpers as original GO validator
# ==============================
def format_terms_for_prompt(go_terms):
    formatted = []
    for go_id in go_terms:
        go_id = go_id.upper()
        if go_id not in godag:
            continue
        term          = godag[go_id]
        name          = getattr(term, "name", "NA")
        definition    = getattr(term, "defn", "NA")
        namespace     = getattr(term, "namespace", "NA")
        parents       = [p.id for p in getattr(term, "parents", []) if hasattr(p, "id")]
        synonyms      = [s.text for s in getattr(term, "synonym", [])]
        relationships = {}
        if hasattr(term, "relationship"):
            for rel_type, go_set in term.relationship.items():
                relationships[rel_type] = [
                    t.item_id if hasattr(t, "item_id") else str(t) for t in go_set
                ]
        formatted.append(f"""
ID: {go_id}
Name: {name}
Namespace: {namespace}
Definition: {definition}
is_a Parents: {", ".join(parents) if parents else "NA"}
Synonyms: {", ".join(synonyms) if synonyms else "NA"}
Relationships: {json.dumps(relationships) if relationships else "NA"}
""")
    return "\n".join(formatted)


def build_prompt(go_terms):
    valid_terms = [t.upper() for t in go_terms if t.upper() in godag]
    terms_text  = format_terms_for_prompt(valid_terms)
    separator   = "-" * 40
    prompt = f"""
You are a molecular biology ontology expert.

Analyze the following Gene Ontology (GO) terms and complete all 4 tasks.

TASKS:
Task 1. Determine if the community is biologically coherent (true/false).
Task 2. Write the common biological theme in ONE sentence.
Task 3. Score coherence from 1 to 10 using this rubric:
    10  → All terms belong to the same biological mechanism or pathway
    8-9 → Strong functional relationship across all terms
    6-7 → Same biological domain but some terms have different roles
    4-5 → Loosely related, weak biological context
    2-3 → Mostly unrelated terms
    1   → Completely unrelated terms
Task 4. Identify outlier terms whose removal would increase overall coherence.
         If no outliers exist, return an empty list.

RULES:
- Output ONLY valid JSON, no explanation outside JSON.
- "outliers" must only contain GO IDs from the input list.
- "outliers" can be empty if all terms fit the theme.

Output format:
{{
  "reasoning": "brief reasoning about coherence",
  "coherent": true or false,
  "theme": "one sentence biological theme",
  "score": integer between 1 and 10,
  "outliers": ["GO:XXXX", "GO:YYYY"]
}}

Community GO terms:
{separator}
{terms_text}
{separator}
"""
    return prompt, valid_terms


def call_llm(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0}},
                timeout=120
            )
            if response.status_code != 200:
                print(f"    [!] HTTP {response.status_code}, attempt {attempt+1}")
                continue
            text = response.json().get("response", "").strip()
            if "```" in text:
                parts = text.split("```")
                if len(parts) > 1:
                    text = parts[1]
                    if text.startswith("json"):
                        text = text[4:]
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            print(f"    [!] JSON parse error (attempt {attempt+1}): {e}")
        except Exception as e:
            print(f"    [!] LLM error (attempt {attempt+1}): {e}")
    return None


def validate_outliers(llm_result, input_go_terms):
    if llm_result is None:
        return None
    valid_ids = set(t.upper() for t in input_go_terms)
    seen, outliers = set(), []
    for t in llm_result.get("outliers", []):
        t = t.upper()
        if t in valid_ids and t not in seen:
            outliers.append(t)
            seen.add(t)
    if len(outliers) >= len(valid_ids):
        outliers = []
    llm_result["outliers"] = outliers
    return llm_result


def score_community(terms):
    valid_terms = [t.upper() for t in terms if t.upper() in godag]
    if len(valid_terms) < 2:
        return None
    prompt, valid_terms = build_prompt(valid_terms)
    result = call_llm(prompt)
    result = validate_outliers(result, valid_terms)
    if result is None:
        return None
    return max(1, min(10, int(result.get("score", 1))))


def get_top_n_communities(node, n=TOP_N):
    comm_counts = defaultdict(int)
    for neighbor in edge_map.get(node.upper(), set()):
        c = node_to_comm.get(neighbor.upper())
        if c:
            comm_counts[c] += 1
    if not comm_counts:
        print(f"    [debug] {node}: neighbors={len(edge_map.get(node.upper(), set()))}, "
              f"none map to any community")
    return [c for c, _ in sorted(comm_counts.items(), key=lambda x: x[1], reverse=True)[:n]]

# ==============================
# Step 7: Reassign loop
# ==============================
moves           = []
reassigned      = 0
kept_no_conn    = 0
kept_score_drop = 0

for node, original_comm in outliers_to_reassign:

    top_comms = get_top_n_communities(node)

    if not top_comms:
        print(f"  {node} : kept (no connections to any community)")
        kept_no_conn += 1
        continue

    assigned = False

    for candidate_comm in top_comms:

        score_before = comm_scores.get(candidate_comm)
        if score_before is None:
            score_before = score_community(comm_nodes[candidate_comm])
        if score_before is None:
            print(f"  {node} : {candidate_comm} skipped (no valid GO terms to score)")
            continue

        # add outlier to candidate and score
        comm_nodes[candidate_comm].append(node)
        score_after = score_community(comm_nodes[candidate_comm])

        if score_after is not None and score_after >= score_before:
            # accept
            node_to_comm[node]           = candidate_comm
            comm_scores[candidate_comm]  = score_after
            print(f"  {node} : {original_comm} -> {candidate_comm} "
                  f"(score {score_before} -> {score_after}) ✓")
            moves.append({
                "node"          : node,
                "from_community": original_comm,
                "to_community"  : candidate_comm,
                "score_before"  : score_before,
                "score_after"   : score_after
            })
            reassigned += 1
            assigned = True
            break
        else:
            # remove and try next
            comm_nodes[candidate_comm].remove(node)
            print(f"  {node} : {candidate_comm} rejected "
                  f"(score {score_before} -> {score_after}) ✗")

    if not assigned:
        print(f"  {node} : kept in {original_comm} (no candidate improved/maintained score)")
        kept_score_drop += 1

# ==============================
# Step 8: Save
# ==============================
with open(OUTPUT_FILE, "w") as f:
    json.dump(moves, f, indent=2)

print(f"\nReassigned        : {reassigned}")
print(f"Kept (no conn)    : {kept_no_conn}")
print(f"Kept (score drop) : {kept_score_drop}")
print(f"Saved             : {OUTPUT_FILE}")