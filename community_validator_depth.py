import json
import requests
import numpy as np
from itertools import combinations
from goatools.base import get_godag
from goatools.semantic import semantic_similarity

# ==============================
# CONFIG
# ==============================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

TEST_MODE = False
TEST_SIZE = 10
COMMUNITIES_FILE = "communities_se2/level_2_communities.json"
GO_OBO_FILE = "go-basic.obo"
OUTPUT_FILE = "llm_judge_se2_results_level2.json"

# ==============================
# LOAD DATA
# ==============================

print("Loading GO DAG...")
godag = get_godag(GO_OBO_FILE, optional_attrs={"def", "synonym", "xref", "property_value"})
print(f"✓ Loaded {len(godag)} GO terms")

print("Loading communities...")
with open(COMMUNITIES_FILE, "r") as f:
    communities = json.load(f)

community_items = list(communities.items())

if TEST_MODE:
    community_items = community_items[:TEST_SIZE]
    print(f"TEST_MODE: {len(community_items)} communities")

# ==============================
# HELPERS
# ==============================

def format_terms_for_prompt(go_terms):
    formatted = []
    for go_id in go_terms:
        go_id = go_id.upper()
        if go_id not in godag:
            continue
        term = godag[go_id]

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
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0}
                },
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


def compute_lin_score(go_terms):
    """Mean pairwise Lin semantic similarity for a set of GO terms (0.0–1.0)."""
    sims = []
    for t1, t2 in combinations(go_terms, 2):
        s = semantic_similarity(t1, t2, godag)
        if s is not None:
            sims.append(s)
    return round(float(np.mean(sims)), 4) if sims else None


def validate_outliers(llm_result, input_go_terms):
    """
    Guarantee:
    - outliers is a strict subset of input_go_terms
    - no duplicates in outliers
    - outliers + cleaned_terms = input_go_terms (enforced after this call)
    """
    if llm_result is None:
        return None

    valid_ids = set(t.upper() for t in input_go_terms)

    # Strip hallucinated IDs + deduplicate
    seen     = set()
    outliers = []
    for t in llm_result.get("outliers", []):
        t = t.upper()
        if t in valid_ids and t not in seen:
            outliers.append(t)
            seen.add(t)

    # If LLM flagged everything, trust nothing
    if len(outliers) >= len(valid_ids):
        outliers = []

    llm_result["outliers"] = outliers

    # Hard guarantees
    assert set(outliers).issubset(valid_ids),    "outliers must be subset of input"
    assert len(set(outliers)) == len(outliers),  "no duplicates in outliers"

    return llm_result


# ==============================
# MAIN PIPELINE
# ==============================

# Load existing results to support resume after crash
import os
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        results = json.load(f)
    done_ids = {r["community_id"] for r in results}
    print(f"Resuming: {len(done_ids)} communities already done, skipping them.")
else:
    results  = []
    done_ids = set()

print(f"\nProcessing {len(community_items)} communities...\n")
print("=" * 60)

for idx, (comm_id, terms) in enumerate(community_items):

    if comm_id in done_ids:
        continue

    go_terms = [
        t.upper() for t in terms
        if t.lower().startswith("go:") and t.upper() in godag
    ]

    print(f"\n[{idx+1}/{len(community_items)}] Community : {comm_id}")
    print(f"  Total valid GO terms : {len(go_terms)}")

    if len(go_terms) < 2:
        print(f"  Status               : SKIPPED (fewer than 2 valid terms)")
        print("-" * 60)
        continue

    # ── First judge ──
    prompt, valid_terms = build_prompt(go_terms)
    result              = call_llm(prompt)
    result              = validate_outliers(result, valid_terms)

    if result is None:
        print(f"  Status               : FAILED (LLM error after retries)")
        print("-" * 60)
        results.append({
            "community_id"  : comm_id,
            "original_terms": go_terms,
            "status"        : "failed"
        })
        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)
        continue

    initial_score = max(1, min(10, int(result.get("score", 1))))
    outliers      = result["outliers"]

    # Every term is in exactly one place
    cleaned_terms = [t for t in go_terms if t not in set(outliers)]
    assert len(outliers) + len(cleaned_terms) == len(go_terms)

    print(f"  Initial score        : {initial_score}/10")
    print(f"  Coherent             : {result.get('coherent')}")
    print(f"  Theme                : {result.get('theme', '')}")
    print(f"  Reasoning            : {result.get('reasoning', '')}")
    print(f"  Outliers found       : {outliers if outliers else 'None'}")

    # ── Rejudge without outliers if any found ──
    final_score        = initial_score
    final_theme        = result.get("theme", "")
    final_reasoning    = result.get("reasoning", "")
    final_coherent     = result.get("coherent", False)
    confirmed_outliers = []

    if outliers and len(cleaned_terms) >= 2:

        print(f"\n  Rejudging without outliers...")

        prompt2, valid_terms2 = build_prompt(cleaned_terms)
        result2               = call_llm(prompt2)
        result2               = validate_outliers(result2, valid_terms2)

        if result2:
            new_score = max(1, min(10, int(result2.get("score", 1))))

            print(f"  Score after removal  : {new_score}/10")
            print(f"  Theme after removal  : {result2.get('theme', '')}")
            print(f"  Reasoning            : {result2.get('reasoning', '')}")

            if new_score > initial_score:
                final_score        = new_score
                final_theme        = result2.get("theme", "")
                final_reasoning    = result2.get("reasoning", "")
                final_coherent     = result2.get("coherent", False)
                confirmed_outliers = outliers
                print(f"  Outlier verdict      : CONFIRMED ✓ ({initial_score} → {new_score})")
            else:
                print(f"  Outlier verdict      : NOT CONFIRMED ✗ (score did not improve)")
        else:
            print(f"  Rejudge              : FAILED (LLM error)")

    # If outliers not confirmed, cleaned_terms = full original input
    if not confirmed_outliers:
        cleaned_terms = go_terms

    # Final guarantee
    assert len(confirmed_outliers) + len(cleaned_terms) == len(go_terms)

    lin_score = compute_lin_score(cleaned_terms)

    print(f"\n  ── Final Result ──")
    print(f"  Final score          : {final_score}/10")
    print(f"  Lin similarity       : {lin_score if lin_score is not None else 'N/A'}")
    print(f"  Confirmed outliers   : {confirmed_outliers if confirmed_outliers else 'None'}")
    print(f"  Cleaned terms        : {cleaned_terms}")
    print("-" * 60)

    results.append({
        "community_id"  : comm_id,
        "status"        : "ok",
        "original_terms": go_terms,
        "initial_score" : initial_score,
        "final_score"   : final_score,
        "lin_score"     : lin_score,
        "score_improved": final_score > initial_score,
        "coherent"      : final_coherent,
        "theme"         : final_theme,
        "reasoning"     : final_reasoning,
        "outliers"      : confirmed_outliers,
        "cleaned_terms" : cleaned_terms
    })
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

# ==============================
# SUMMARY
# ==============================

total      = len(results)
failed     = sum(1 for r in results if r["status"] == "failed")
improved   = sum(1 for r in results if r.get("score_improved"))
ok_results = [r for r in results if r["status"] == "ok"]
scores     = [r["final_score"] for r in ok_results]
lin_scores = [r["lin_score"] for r in ok_results if r["lin_score"] is not None]
mean_score = sum(scores) / len(scores) if scores else 0
mean_lin   = sum(lin_scores) / len(lin_scores) if lin_scores else 0
high_coh   = sum(1 for s in scores if s >= 7)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Total processed  : {total}")
print(f"  Failed           : {failed}")
print(f"  Outliers removed : {improved}")
print(f"  Mean LLM score   : {mean_score:.2f}/10")
print(f"  Mean Lin score   : {mean_lin:.4f}  (0=unrelated, 1=identical)")
print(f"  High coherence   : {high_coh}/{len(scores)} ({100*high_coh/len(scores):.1f}%)")
print(f"  Saved            : {OUTPUT_FILE}")
print("=" * 60)