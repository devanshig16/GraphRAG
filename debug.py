import pickle
import json

# Load data
with open("giant_component_directed.pkl", 'rb') as f:
    G = pickle.load(f)

with open("communities_4/level_2_communities.json") as f:
    communities = json.load(f)

# Pick your community
cid = "0.0.0"
nodes = set(communities[cid])

# Print relationship table
print(f"Community {cid} — {len(nodes)} nodes\n")
print(f"{'ID':<5} {'SOURCE':<40} {'TARGET':<40} {'RELATIONSHIP'}")
print("-" * 120)

rel_id = 0
for u, v, data in G.edges(data=True):
    if u in nodes and v in nodes:
        rel = data.get('relationship', 'N/A')
        print(f"{rel_id:<5} {u:<40} {v:<40} {rel}")
        rel_id += 1

print(f"\nTotal edges: {rel_id}")