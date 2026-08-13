import json
import random
import requests
from collections import defaultdict
from goatools.base import get_godag

# ==============================
# CONFIG
# ==============================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

TEST_MODE = False
TEST_SIZE = 100
COMMUNITIES_FILE = "communities_4/level_2_communities.json"
GO_OBO_FILE = "go-basic.obo"

INJECT_INTRUDER = False
STABILITY_RUNS = 1

# ==============================
# LOAD DATA
# ==============================

print("Loading GO DAG...")

godag = get_godag(
    GO_OBO_FILE,
    optional_attrs={"def", "synonym", "xref", "property_value"}
)

print(f"✓ Loaded {len(godag)} GO terms")

print("Loading communities...")

with open(COMMUNITIES_FILE, "r") as f:
    communities = json.load(f)

community_items = list(communities.items())

if TEST_MODE:
    community_items = community_items[:TEST_SIZE]
    print(f"Running TEST_MODE on {len(community_items)} communities")

# Global GO pool for intruder injection
all_go_ids = {
    t.upper()
    for terms in communities.values()
    for t in terms
    if t.lower().startswith("go:")
}


def extract_term_metadata(term):

    definition = getattr(term, "defn", "NA")
    namespace = getattr(term, "namespace", "NA")

    # is_a parents
    parents = []
    if hasattr(term, "parents"):
        parents = [p.id for p in term.parents if hasattr(p, "id")]

    # synonyms
    synonyms = []
    if hasattr(term, "synonym"):
        synonyms = [s.text for s in term.synonym]

    # xref
    xrefs = list(getattr(term, "xref", set()))

    # relationships (part_of, regulates etc.)
    relationships = {}

    if hasattr(term, "relationship"):
        for rel_type, go_set in term.relationship.items():
            relationships[rel_type] = list(go_set)

    return {
        "definition": definition,
        "namespace": namespace,
        "parents": parents,
        "synonyms": synonyms,
        "xref": xrefs,
        "relationships": relationships
    }

# ==============================
# PROMPT BUILDER
# ==============================

def build_prompt(go_terms):

    formatted_terms = []

    for go_id in go_terms:

        go_id = go_id.upper()

        if go_id not in godag:
            continue

        term = godag[go_id]

        # Basic ontology info
        name = getattr(term, "name", "NA")
        definition = getattr(term, "defn", "NA")
        namespace = getattr(term, "namespace", "NA")

        # ---- is_a parents ----
        parents = []
        if hasattr(term, "parents"):
            parents = [p.id for p in term.parents if hasattr(p, "id")]

        # ---- synonyms ----
        synonyms = []
        if hasattr(term, "synonym"):
            synonyms = [s.text for s in term.synonym]

        # ---- xref ----
        xrefs = list(getattr(term, "xref", set()))

        # ---- relationships ----
        relationships = {}
        if hasattr(term, "relationship"):
            for rel_type, go_set in term.relationship.items():
                relationships[rel_type] = list(go_set)

        formatted_terms.append(f"""
ID: {go_id}
Name: {name}

Namespace: {namespace}
Definition: {definition}

is_a Parents: {", ".join(parents) if parents else "NA"}

Synonyms: {", ".join(synonyms) if synonyms else "NA"}

Cross References: {", ".join(xrefs) if xrefs else "NA"}

Relationships: {json.dumps(relationships) if relationships else "NA"}
""")

    separator = "-" * 40
    terms_text = "\n".join(formatted_terms)
    

    prompt = f"""
You are a molecular biology ontology expert.

Your TASKS are:
Task 1. Determine if the community is biologically coherent.
Task 2. Write the common biological theme in ONE sentence.
Task 3. Score coherence from 1 to 10.

Scoring guideline for coherence score:
10 → Same biological mechanism or pathway
8-9 → Strong functional biological relation
6-7 → Same biological domain but different roles
4-5 → Loose biological context
2-3 → Weak similarity
1 → Unrelated terms

Task 4. Identify all potential outlier terms in the community, such that removing them would increase the overall coherence of the community. 

IMPORTANT RULES:
- Output MUST be valid JSON.
- Output ONLY JSON.
- Theme must be non-empty.
- All communities should either belong to outliers or non-outliers and they are mutually exclusive.
- non-outliers cann't be empty.

Output format:
{{
  "coherent": true/false,
  "theme": "one sentence theme",
  "score": integer between 1 and 10,
  "outliers": ["GO:XXXX"],
  "Non-Outliers": ["GO:YYYY"]
}}

Community GO terms:
{separator}
{terms_text}
{separator}
"""

    return prompt, go_terms


# ==============================
# LLM CALL
# ==============================

def call_llm(prompt):

    for attempt in range(2):

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0}
                },
                timeout=60
            )

            if response.status_code != 200:
                print(f"HTTP {response.status_code}")
                continue

            text = response.json().get("response", "").strip()

            # Remove markdown fences if present
            if "```" in text:
                parts = text.split("```")
                if len(parts) > 1:
                    text = parts[1]
                    if text.startswith("json"):
                        text = text[4:]

            return json.loads(text.strip())

        except json.JSONDecodeError as e:
            print("JSON parse error:", e)

        except Exception as e:
            print("LLM call error:", e)

    return None


# ==============================
# EXPERIMENT
# ==============================

results = []
community_scores = defaultdict(list)

print("\nProcessing communities...")

for comm_id, terms in community_items:

    go_terms = [
        t.upper()
        for t in terms
        if t.lower().startswith("go:")
    ]

    # Validate ontology presence
    go_terms = [g for g in go_terms if g in godag]

    if len(go_terms) < 2:
        continue

    injected_term = None

    if INJECT_INTRUDER:
        candidates = list(all_go_ids - set(go_terms))
        if candidates:
            injected_term = random.choice(candidates)
            go_terms.append(injected_term)

    for run in range(STABILITY_RUNS):

        prompt, filtered_terms = build_prompt(go_terms)

        if len(filtered_terms) < 2:
            continue

        llm_result = call_llm(prompt)

        if llm_result is None:
            continue

        score = max(1, min(10, int(llm_result.get("score", 0))))

        community_scores[comm_id].append(score)

        results.append({
            "community_id": comm_id,
            "run": run,
            "score": score,
            "coherent": llm_result.get("coherent"),
            "theme": llm_result.get("theme", ""),
            "outliers": llm_result.get("outliers", []),
            "non_outliers": llm_result.get("Non-Outliers", []),
            "injected_term": injected_term
        })


# ==============================
# AGGREGATION
# ==============================

if community_scores:

    community_avg_scores = [
        sum(v) / len(v)
        for v in community_scores.values()
        if len(v) > 0
    ]

    mean_score = sum(community_avg_scores) / len(community_avg_scores)

    coherent_pct = (
        sum(1 for s in community_avg_scores if s > 5)
        / len(community_avg_scores)
    ) * 100

    print("\n==============================")
    print(f"Total communities processed: {len(community_avg_scores)}")
    print("==============================")
    print("FINAL RESULTS")
    print("==============================")
    print(f"Mean coherence score: {mean_score:.2f}")
    print(f"% High coherence communities: {coherent_pct:.2f}")


# ==============================
# SAVE OUTPUT
# ==============================

# with open("llm_judge_results_level2.json", "w") as f:
#     json.dump(results, f, indent=2)

# print("\n✓ Results saved to llm_judge_results_level2.json")