import networkx as nx

def load_triplets(filepath):
    """Load triplets into NetworkX graph"""
    G = nx.Graph()
    triplet_count = 0
    
    print("Loading triplets...")
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Parse: subject | relationship | object
            parts = [p.strip() for p in line.split('|')]
            
            if len(parts) == 3:
                subject, relation, obj = parts
                G.add_edge(subject, obj, relationship=relation)
                triplet_count += 1
    
    print(f"✓ Loaded {triplet_count:,} triplets")
    return G

# Load your data
G = load_triplets('triplets_output_part.txt')

# Print basic statistics
print("\n" + "="*60)
print("KNOWLEDGE GRAPH STATISTICS")
print("="*60)
print(f"Total Entities (Nodes):      {G.number_of_nodes():,}")
print(f"Total Relationships (Edges): {G.number_of_edges():,}")
print(f"Average connections per entity: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
print("="*60)

import networkx as nx
from collections import defaultdict

def load_triplets(filepath):
    """Load triplets into NetworkX graph"""
    G = nx.Graph()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 3:
                subject, relation, obj = parts
                G.add_edge(subject, obj, relationship=relation)
    
    return G

# Load the graph
print("Loading graph...")
G = load_triplets('triplets_output_part.txt')
print("✓ Graph loaded\n")

# ============================================================
# ANALYSIS 1: Relationship Types
# ============================================================
print("="*60)
print("RELATIONSHIP TYPES IN YOUR KNOWLEDGE GRAPH")
print("="*60)

relationship_types = defaultdict(int)
for u, v, data in G.edges(data=True):
    relationship_types[data['relationship']] += 1

print(f"\nTotal unique relationship types: {len(relationship_types)}\n")

print("Top 15 most common relationships:")
for i, (rel, count) in enumerate(sorted(relationship_types.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:15], 1):
    print(f"{i:2}. {rel:40} {count:6,} times")

# ============================================================
# ANALYSIS 2: Hub Entities (Most Connected)
# ============================================================
print("\n" + "="*60)
print("TOP HUB ENTITIES (Most Connected Nodes)")
print("="*60)

# Calculate degree for each node
node_degrees = dict(G.degree())
top_hubs = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:20]

print("\nTop 20 most connected entities:")
for i, (node, degree) in enumerate(top_hubs, 1):
    print(f"{i:2}. {node:45} {degree:4} connections")

# ============================================================
# ANALYSIS 3: Graph Connectivity
# ============================================================
print("\n" + "="*60)
print("GRAPH CONNECTIVITY")
print("="*60)

# Check if graph is connected
is_connected = nx.is_connected(G)
print(f"\nIs graph fully connected? {is_connected}")

# Find connected components
components = list(nx.connected_components(G))
print(f"Number of separate components: {len(components)}")

if len(components) > 1:
    component_sizes = sorted([len(c) for c in components], reverse=True)
    print(f"\nComponent sizes:")
    print(f"  Largest component: {component_sizes[0]:,} nodes ({component_sizes[0]/G.number_of_nodes()*100:.1f}%)")
    if len(component_sizes) > 1:
        print(f"  2nd largest: {component_sizes[1]:,} nodes")
        print(component_sizes[:9])
        print(f"  Smallest components: {component_sizes[-5:]} nodes")


print("\n" + "="*60)
print("✓ ANALYSIS COMPLETE")
print("="*60)