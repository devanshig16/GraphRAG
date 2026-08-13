import networkx as nx
import pickle
import json

print("="*60)
print("COMMUNITY VERIFICATION")
print("="*60)

# Load graph
with open('giant_component.pkl', 'rb') as f:
    G = pickle.load(f)

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Check all 3 levels
for level in [0, 1, 2]:
    print(f"\n{'='*60}")
    print(f"LEVEL {level}")
    print(f"{'='*60}")
    
    # Load communities
    with open(f'communities_4/level_{level}_communities.json', 'r') as f:
        communities = json.load(f)
    
    # 1. MODULARITY
    community_list = list(communities.values())
    modularity = nx.community.modularity(G, community_list)
    
    print(f"\n1. MODULARITY: {modularity:.4f}")
    if modularity > 0.3:
        print("   ✓ Good communities")
    else:
        print("   ✗ Weak communities")
    
    # 2. COVERAGE
    all_nodes = set()
    for members in communities.values():
        all_nodes.update(members)
    
    coverage = len(all_nodes)
    coverage_pct = (coverage / G.number_of_nodes()) * 100
    
    print(f"\n2. COVERAGE: {coverage}/{G.number_of_nodes()} nodes ({coverage_pct:.1f}%)")
    if coverage == G.number_of_nodes():
        print("   ✓ All nodes assigned")
    else:
        missing = G.number_of_nodes() - coverage
        print(f"   ✗ Missing {missing} nodes")

print(f"\n{'='*60}")
print("VERIFICATION COMPLETE")
print(f"{'='*60}")