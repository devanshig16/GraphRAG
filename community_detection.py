# import networkx as nx
# import pickle
# import json
# import os
# from collections import defaultdict
# import igraph as ig
# import leidenalg
# import numpy as np

# print("="*60)
# print("STEP 2: RECURSIVE HIERARCHICAL LEIDEN")
# print("="*60)

# # Create output directory
# os.makedirs('communities_4', exist_ok=True)

# # ============================================================
# # CONFIGURATION
# # ============================================================

# MAX_CLUSTER_SIZE = 20  # communities larger than this get recursed
# RESOLUTION = 1.0       # higher = more/smaller communities, lower = fewer/larger

# # ============================================================
# # STEP 1: Load Giant Component
# # ============================================================
# print("\nLoading giant component...")
# with open('giant_component.pkl', 'rb') as f:
#     G_nx = pickle.load(f)

# print(f"✓ Graph: {G_nx.number_of_nodes():,} nodes, {G_nx.number_of_edges():,} edges")

# # ============================================================
# # STEP 2: Convert to igraph
# # ============================================================
# print("\nConverting to igraph format...")

# node_list = list(G_nx.nodes())
# node_to_idx = {node: idx for idx, node in enumerate(node_list)}

# g = ig.Graph()
# g.add_vertices(len(node_list))
# edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_nx.edges()]
# g.add_edges(edges)

# print(f"✓ igraph: {g.vcount()} vertices, {g.ecount()} edges")

# # ============================================================
# # STEP 3: Recursive Leiden Function
# # ============================================================

# def recursive_leiden_igraph(graph, node_indices, max_size=MAX_CLUSTER_SIZE,
#                              current_level=0, parent_id="", max_levels=10):
#     """
#     Recursively partition graph using Leiden algorithm.
#     Every node is guaranteed an assignment at every level.
#     Stops when community <= max_size or Leiden can't split further.
#     """
#     n_nodes = len(node_indices)
#     indent = "  " * current_level

#     # Leaf: small enough or hit depth limit
#     if n_nodes <= max_size or current_level >= max_levels:
#         comm_id = parent_id if parent_id else "0"
#         print(f"{indent}Level {current_level}: {n_nodes} nodes → LEAF")
#         return {idx: comm_id for idx in node_indices}

#     # Extract subgraph
#     subgraph = graph.subgraph(node_indices)

#     # Handle disconnected subgraph
#     if not subgraph.is_connected():
#         components = subgraph.connected_components()
#         print(f"{indent}Level {current_level}: {n_nodes} nodes, {len(components)} components")
#         all_assignments = {}
#         for comp_idx, component in enumerate(components):
#             comp_nodes = [node_indices[i] for i in component]
#             comp_id = f"{parent_id}.{comp_idx}" if parent_id else str(comp_idx)
#             all_assignments.update(
#                 recursive_leiden_igraph(graph, comp_nodes, max_size,
#                                         current_level, comp_id, max_levels)
#             )
#         return all_assignments
#     RESOLUTION = 1+ current_level*0.5
#     # Run Leiden
#     partition = leidenalg.find_partition(
#         subgraph,
#         leidenalg.RBConfigurationVertexPartition,
#         resolution_parameter=RESOLUTION,
#         seed=42
#     )

#     n_communities = len(partition)
#     print(f"{indent}Level {current_level}: Running Leiden on {n_nodes} nodes...")
#     print(f"{indent}  → Found {n_communities} communities")

#     # Can't split further — natural leaf
#     if n_communities == 1:
#         comm_id = parent_id if parent_id else "0"
#         print(f"{indent}  → Can't split - STOP")
#         return {node_indices[i]: comm_id for i in range(n_nodes)}

#     # Assign communities and recurse large ones
#     node_assignments = {}
#     for comm_idx, members in enumerate(partition):
#         comm_id = f"{parent_id}.{comm_idx}" if parent_id else str(comm_idx)
#         original_nodes = [node_indices[i] for i in members]

#         if len(original_nodes) > max_size:
#             sub_assignments = recursive_leiden_igraph(
#                 graph, original_nodes, max_size,
#                 current_level + 1, comm_id, max_levels
#             )
#             node_assignments.update(sub_assignments)
#         else:
#             for orig_idx in original_nodes:
#                 node_assignments[orig_idx] = comm_id

#     return node_assignments

# # ============================================================
# # STEP 4: Run Recursive Leiden
# # ============================================================

# print("\n" + "="*60)
# print("RUNNING RECURSIVE LEIDEN")
# print("="*60)

# all_node_indices = list(range(len(node_list)))

# hierarchical_assignments = recursive_leiden_igraph(
#     g,
#     all_node_indices,
#     max_size=MAX_CLUSTER_SIZE
# )

# print(f"\n✓ Recursive partitioning complete!")
# print(f"  Total hierarchical assignments: {len(hierarchical_assignments)}")

# missing = set(range(len(node_list))) - set(hierarchical_assignments.keys())
# if missing:
#     print(f"  ⚠ WARNING: {len(missing)} nodes have no assignment!")
# else:
#     print(f"  ✓ All nodes assigned")

# # ============================================================
# # STEP 5: Extract Levels
# # ============================================================

# print("\n" + "="*60)
# print("EXTRACTING HIERARCHY LEVELS")
# print("="*60)

# # Map indices back to node names
# hierarchical_assignments_named = {
#     node_list[idx]: path
#     for idx, path in hierarchical_assignments.items()
# }

# # Find max depth
# max_depth = max(len(path.split('.')) for path in hierarchical_assignments_named.values())
# print(f"\nHierarchy depth: {max_depth} levels")

# # Extract each level
# community_data = {}

# for level_idx in range(max_depth):
#     level_assignments = {}

#     for node, path in hierarchical_assignments_named.items():
#         parts = path.split('.')
#         level_comm = '.'.join(parts[:level_idx + 1]) if level_idx < len(parts) else path
#         level_assignments[node] = level_comm

#     community_data[f"level_{level_idx}"] = level_assignments
#     n_comms = len(set(level_assignments.values()))
#     print(f"  Level {level_idx}: {n_comms} communities")

# # ============================================================
# # STEP 6: Analyze & Save Each Level
# # ============================================================

# print("\n" + "="*60)
# print("ANALYZING & SAVING COMMUNITIES")
# print("="*60)

# statistics = []

# for level_idx in range(max_depth):
#     level_name = f"level_{level_idx}"
#     node_to_comm = community_data[level_name]

#     comm_to_nodes = defaultdict(list)
#     for node, comm in node_to_comm.items():
#         comm_to_nodes[comm].append(node)

#     n_communities = len(comm_to_nodes)
#     sizes = [len(nodes) for nodes in comm_to_nodes.values()]

#     avg_size = np.mean(sizes)
#     min_size = np.min(sizes)
#     max_size = np.max(sizes)

#     level_desc = ['Coarsest', 'Medium', 'Fine', 'Finest'][min(level_idx, 3)]

#     print(f"\n{level_name.upper()} ({level_desc}):")
#     print(f"  Communities: {n_communities}")
#     print(f"  Avg size: {avg_size:.1f} nodes")
#     print(f"  Range: {min_size} - {max_size} nodes")

#     statistics.append({
#         'level': level_idx,
#         'communities': n_communities,
#         'avg_size': float(avg_size),
#         'min_size': int(min_size),
#         'max_size': int(max_size)
#     })

#     filename = f"communities_4/{level_name}_communities.json"
#     with open(filename, 'w') as f:
#         json.dump(dict(comm_to_nodes), f, indent=2)

#     print(f"  ✓ Saved to: {filename}")

# # Save full hierarchy
# with open('communities_4/hierarchy.json', 'w') as f:
#     json.dump(hierarchical_assignments_named, f, indent=2)

# print(f"\n✓ Saved full hierarchy to: communities_4/hierarchy.json")

# # Save statistics
# with open('communities_4/statistics.json', 'w') as f:
#     json.dump({
#         'graph_size': G_nx.number_of_nodes(),
#         'graph_edges': G_nx.number_of_edges(),
#         'hierarchy_depth': max_depth,
#         'levels': statistics
#     }, f, indent=2)

# print(f"✓ Saved statistics to: communities_2/statistics.json")

# # ============================================================
# # STEP 7: Summary
# # ============================================================

# print("\n" + "="*60)
# print("SUMMARY")
# print("="*60)

# print(f"\nGraph: {G_nx.number_of_nodes():,} nodes, {G_nx.number_of_edges():,} edges")
# print(f"Hierarchy depth: {max_depth} levels")
# print(f"\nCommunities per level:")
# for stat in statistics:
#     print(f"  Level {stat['level']}: {stat['communities']:4} communities " +
#           f"(avg {stat['avg_size']:.1f} nodes, range {stat['min_size']}-{stat['max_size']})")

# print(f"\nFiles created in communities_2/:")
# print(f"  - hierarchy.json (full hierarchical structure)")
# print(f"  - statistics.json (summary statistics)")
# for level_idx in range(max_depth):
#     print(f"  - level_{level_idx}_communities.json")

# print("\n" + "="*60)
# print("✓ STEP 2 COMPLETE")
# print("="*60)
# print("\nNext: Community summarization with LLM")

import networkx as nx
import pickle
import json
import os
from collections import defaultdict
import igraph as ig
import leidenalg
import numpy as np

print("="*60)
print("STEP 2: RECURSIVE HIERARCHICAL LEIDEN (FIXED VERSION)")
print("="*60)

# Create output directory
os.makedirs('communities_4', exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

MAX_CLUSTER_SIZE = 40
RESOLUTION = 1.0
MAX_LEVELS = 10

# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading giant component...")
with open('giant_component.pkl', 'rb') as f:
    G_nx = pickle.load(f)

print(f"✓ Graph: {G_nx.number_of_nodes():,} nodes, {G_nx.number_of_edges():,} edges")

# ============================================================
# Convert to igraph
# ============================================================

print("\nConverting to igraph format...")

node_list = list(G_nx.nodes())
node_to_idx = {node: idx for idx, node in enumerate(node_list)}

g = ig.Graph()
g.add_vertices(len(node_list))

edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_nx.edges()]
g.add_edges(edges)

print(f"✓ igraph: {g.vcount()} vertices, {g.ecount()} edges")

# ============================================================
# Recursive Leiden Function (FIXED)
# ============================================================

def recursive_leiden_igraph(graph, node_indices,
                             max_size=MAX_CLUSTER_SIZE,
                             current_level=0,
                             parent_id="",
                             max_levels=MAX_LEVELS,
                             resolution=RESOLUTION):

    n_nodes = len(node_indices)
    indent = "  " * current_level

    # -----------------------------
    # Leaf stopping conditions
    # -----------------------------
    if (
        n_nodes <= max_size
        or current_level >= max_levels
        or n_nodes <= 2
    ):
        comm_id = parent_id if parent_id else "0"
        print(f"{indent}Level {current_level}: {n_nodes} nodes → LEAF")
        return {idx: comm_id for idx in node_indices}

    # -----------------------------
    # Extract subgraph
    # -----------------------------
    subgraph = graph.subgraph(node_indices)

    if not subgraph.is_connected():
        components = subgraph.connected_components()
        print(f"{indent}Level {current_level}: {n_nodes} nodes, {len(components)} components")

        all_assignments = {}

        for comp_idx, component in enumerate(components):
            comp_nodes = [node_indices[i] for i in component]
            comp_id = f"{parent_id}.{comp_idx}" if parent_id else str(comp_idx)

            all_assignments.update(
                recursive_leiden_igraph(
                    graph,
                    comp_nodes,
                    max_size,
                    current_level,
                    comp_id,
                    max_levels,
                    resolution
                )
            )

        return all_assignments

    # -----------------------------
    # Leiden clustering
    # -----------------------------

    partition = leidenalg.find_partition(
        subgraph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=42
    )

    n_communities = len(partition)

    # Stop if cannot split or modularity is weak
    if n_communities == 1 or subgraph.modularity(partition) < 0.1:
        comm_id = parent_id if parent_id else "0"
        print(f"{indent}Level {current_level}: {n_nodes} nodes → STOP SPLIT")
        return {node_indices[i]: comm_id for i in range(n_nodes)}

    print(f"{indent}Level {current_level}: Running Leiden on {n_nodes} nodes...")
    print(f"{indent}  → Found {n_communities} communities")

    # -----------------------------
    # Assign communities + recurse
    # -----------------------------

    node_assignments = {}

    for comm_idx, members in enumerate(partition):

        comm_id = f"{parent_id}.{comm_idx}" if parent_id else str(comm_idx)

        original_nodes = [node_indices[i] for i in members]

        if len(original_nodes) > max_size:
            sub_assignments = recursive_leiden_igraph(
                graph,
                original_nodes,
                max_size,
                current_level + 1,
                comm_id,
                max_levels,
                resolution + current_level * 0.2
            )

            node_assignments.update(sub_assignments)

        else:
            for orig_idx in original_nodes:
                node_assignments[orig_idx] = comm_id

    return node_assignments


# ============================================================
# RUN RECURSIVE LEIDEN
# ============================================================

print("\n" + "="*60)
print("RUNNING RECURSIVE LEIDEN")
print("="*60)

all_node_indices = list(range(len(node_list)))

hierarchical_assignments = recursive_leiden_igraph(
    g,
    all_node_indices
)

print(f"\n✓ Recursive partitioning complete!")
print(f"  Total hierarchical assignments: {len(hierarchical_assignments)}")

# Check missing nodes
missing = set(range(len(node_list))) - set(hierarchical_assignments.keys())

if missing:
    print(f"⚠ WARNING: {len(missing)} nodes have no assignment!")
else:
    print(f"✓ All nodes assigned")

# ============================================================
# Extract Named Hierarchy
# ============================================================

hierarchical_assignments_named = {
    node_list[idx]: path
    for idx, path in hierarchical_assignments.items()
}

# Compute hierarchy depth
max_depth = max(len(path.split('.')) for path in hierarchical_assignments_named.values())

print(f"\nHierarchy depth: {max_depth} levels")

# ============================================================
# Extract Levels
# ============================================================

community_data = {}

for level_idx in range(max_depth):

    level_assignments = {}

    for node, path in hierarchical_assignments_named.items():

        parts = path.split('.')
        level_comm = '.'.join(parts[:level_idx + 1])

        level_assignments[node] = level_comm

    community_data[f"level_{level_idx}"] = level_assignments

    n_comms = len(set(level_assignments.values()))
    print(f"  Level {level_idx}: {n_comms} communities")

# ============================================================
# SAVE COMMUNITIES
# ============================================================

statistics = []

for level_idx in range(max_depth):

    level_name = f"level_{level_idx}"
    node_to_comm = community_data[level_name]

    comm_to_nodes = defaultdict(list)

    for node, comm in node_to_comm.items():
        comm_to_nodes[comm].append(node)

    sizes = [len(nodes) for nodes in comm_to_nodes.values()]

    avg_size = np.mean(sizes)
    min_size = np.min(sizes)
    max_size = np.max(sizes)

    print(f"\n{level_name.upper()}:")
    print(f"  Communities: {len(comm_to_nodes)}")
    print(f"  Avg size: {avg_size:.1f}")
    print(f"  Range: {min_size}-{max_size}")

    statistics.append({
        'level': level_idx,
        'communities': len(comm_to_nodes),
        'avg_size': float(avg_size),
        'min_size': int(min_size),
        'max_size': int(max_size)
    })

    filename = f"communities_4/{level_name}_communities.json"

    with open(filename, 'w') as f:
        json.dump(dict(comm_to_nodes), f, indent=2)

    print(f"  ✓ Saved → {filename}")

# ============================================================
# Save hierarchy + statistics
# ============================================================

with open('communities_4/hierarchy.json', 'w') as f:
    json.dump(hierarchical_assignments_named, f, indent=2)

with open('communities_4/statistics.json', 'w') as f:
    json.dump({
        'graph_size': G_nx.number_of_nodes(),
        'graph_edges': G_nx.number_of_edges(),
        'hierarchy_depth': max_depth,
        'levels': statistics
    }, f, indent=2)

print("\n✓ STEP 2 COMPLETE")
print("="*60)