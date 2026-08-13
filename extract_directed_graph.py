import networkx as nx
import pickle

def load_triplets(filepath):
    """Load triplets into NetworkX graph"""
    G = nx.DiGraph()
    
    print("Loading triplets...")
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 3:
                subject, relation, obj = parts
                G.add_edge(subject, obj, relationship=relation)
    
    print(f"✓ Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G

# ============================================================
# STEP 1: Load Full Graph
# ============================================================
print("="*60)
print("STEP 1: EXTRACT GIANT COMPONENT")
print("="*60)

G = load_triplets('triplets_output_part.txt')

# ============================================================
# STEP 2: Find Connected Components
# ============================================================
print("\nFinding connected components...")
components = list(nx.weakly_connected_components(G))
component_sizes = [len(c) for c in components]
component_sizes.sort(reverse=True)

print(f"✓ Found {len(components):,} connected components")
print(f"\nTop 10 component sizes:")
for i, size in enumerate(component_sizes[:10], 1):
    percentage = (size / G.number_of_nodes()) * 100
    print(f"  {i:2}. {size:6,} nodes ({percentage:5.2f}%)")

# ============================================================
# STEP 3: Extract Giant Component
# ============================================================
print("\nExtracting giant component...")

# Get the largest component
giant_component = max(components, key=len)

# Create subgraph
G_giant = G.subgraph(giant_component).copy()

print(f"✓ Giant component extracted")
print(f"  Nodes: {G_giant.number_of_nodes():,}")
print(f"  Edges: {G_giant.number_of_edges():,}")

# ============================================================
# STEP 4: Statistics Comparison
# ============================================================
print("\n" + "="*60)
print("BEFORE vs AFTER FILTERING")
print("="*60)

print(f"\nOriginal graph:")
print(f"  Nodes: {G.number_of_nodes():,}")
print(f"  Edges: {G.number_of_edges():,}")
print(f"  Components: {len(components):,}")

print(f"\nGiant component only:")
print(f"  Nodes: {G_giant.number_of_nodes():,}")
print(f"  Edges: {G_giant.number_of_edges():,}")
print(f"  Components: 1")

nodes_kept = G_giant.number_of_nodes()
nodes_total = G.number_of_nodes()
percentage_kept = (nodes_kept / nodes_total) * 100

print(f"\nData retained: {nodes_kept:,} / {nodes_total:,} nodes ({percentage_kept:.1f}%)")
print(f"Data discarded: {nodes_total - nodes_kept:,} nodes ({100-percentage_kept:.1f}%)")

# ============================================================
# STEP 5: Save Giant Component
# ============================================================
print("\n" + "="*60)
print("SAVING GIANT COMPONENT")
print("="*60)

# Save as pickle for next step
with open('giant_component_directed.pkl', 'wb') as f:
    pickle.dump(G_giant, f)

print(f"✓ Saved to: giant_component_directed.pkl")

# Also save as edge list (human readable)
nx.write_edgelist(G_giant, 'giant_component_directed_edges.txt', data=['relationship'])
print(f"✓ Saved to: giant_component_directed_edges.txt (human readable)")

print("\n" + "="*60)
print("✓ STEP 1 COMPLETE")
print("="*60)
print("\nGiant component ready for community detection!")
print("Next: Run community detection on giant_component.pkl")