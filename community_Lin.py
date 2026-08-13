import json
import random
import numpy as np
from itertools import combinations
from goatools.base import get_godag
from goatools.semantic import semantic_similarity

print("="*60)
print("GO SEMANTIC SIMILARITY VALIDATION")
print("="*60)

# ------------------------------------------------------------
# Load GO DAG
# ------------------------------------------------------------
print("Loading GO database...")
godag = get_godag("go-basic.obo")
print(f"✓ Loaded {len(godag)} GO terms")

# ------------------------------------------------------------
# Load communities
# ------------------------------------------------------------
print("\nLoading communities...")
with open('communities_4/level_2_communities.json', 'r') as f:
    communities = json.load(f)

community_ids = list(communities.keys())

# ------------------------------------------------------------
# Helper function to extract valid GO terms
# ------------------------------------------------------------
def get_valid_go_terms(members):
    go_terms = []
    for m in members:
        if isinstance(m, str) and m.lower().startswith("go:"):
            go_fmt = "GO:" + m.split(":")[1]
            if go_fmt in godag:
                go_terms.append(go_fmt)
    return go_terms

# ------------------------------------------------------------
# WITHIN-community similarity
# ------------------------------------------------------------
print("\nCalculating WITHIN-community similarity...")

within_scores = []
num_communities = min(50, len(communities))

sampled_communities = random.sample(community_ids, num_communities)

for comm_id in sampled_communities:
    members = communities[comm_id]
    go_terms = get_valid_go_terms(members)

    if len(go_terms) < 2:
        continue

    for go1, go2 in combinations(go_terms, 2):
        sim = semantic_similarity(go1, go2, godag)
        if sim is not None:
            within_scores.append(sim)

print(f"✓ Calculated {len(within_scores)} within-community similarities")

# ------------------------------------------------------------
# BETWEEN-community similarity (MATCH SIZE EXACTLY)
# ------------------------------------------------------------
print("\nCalculating BETWEEN-community similarity...")

between_scores = []
target_samples = len(within_scores)
max_attempts = target_samples * 10  # safety limit
attempts = 0

while len(between_scores) < target_samples and attempts < max_attempts:
    attempts += 1

    # pick two different communities
    comm1_id, comm2_id = random.sample(community_ids, 2)
    comm1 = communities[comm1_id]
    comm2 = communities[comm2_id]

    go1_list = get_valid_go_terms(comm1)
    go2_list = get_valid_go_terms(comm2)

    if not go1_list or not go2_list:
        continue

    go1 = random.choice(go1_list)
    go2 = random.choice(go2_list)

    sim = semantic_similarity(go1, go2, godag)
    if sim is not None:
        between_scores.append(sim)

print(f"✓ Calculated {len(between_scores)} between-community similarities")

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------
print("\n" + "="*60)
print("RESULTS")
print("="*60)

within_avg = np.mean(within_scores) if within_scores else 0
between_avg = np.mean(between_scores) if between_scores else 0
ratio = within_avg / between_avg if between_avg > 0 else float("inf")

print(f"\nWithin-community similarity:  {within_avg:.4f}")
print(f"Between-community similarity: {between_avg:.4f}")
print(f"Ratio (Within/Between):       {ratio:.2f}x")

# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if ratio > 2.0:
    print("🔥 VERY STRONG: Communities are highly semantically coherent")
elif ratio > 1.5:
    print("✓ EXCELLENT: Communities group semantically similar GO terms")
elif ratio > 1.2:
    print("✓ GOOD: Communities show semantic coherence")
elif ratio > 1.0:
    print("⚠ WEAK: Slight semantic grouping detected")
else:
    print("✗ POOR: No semantic grouping detected")

print("\n" + "="*60)
