import pickle
import json
import os
import numpy as np
import igraph as ig
import speakeasy2 as se2
from collections import defaultdict

# ==============================
# CONFIG
# ==============================

MAX_CLUSTER_SIZE = 40
MAX_LEVELS       = 10
OUTPUT_DIR       = "communities_se2"
GRAPH_FILE       = "giant_component.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD GRAPH
# ==============================

print("Loading giant component...")
with open(GRAPH_FILE, "rb") as f:
    G_nx = pickle.load(f)

print(f"✓ Graph: {G_nx.number_of_nodes():,} nodes, {G_nx.number_of_edges():,} edges")

# ==============================
# CONVERT TO IGRAPH
# ==============================

print("Converting to igraph...")

node_list   = list(G_nx.nodes())
node_to_idx = {node: idx for idx, node in enumerate(node_list)}

g = ig.Graph()
g.add_vertices(len(node_list))
g.add_edges([(node_to_idx[u], node_to_idx[v]) for u, v in G_nx.edges()])

print(f"✓ igraph: {g.vcount()} vertices, {g.ecount()} edges")

# ==============================
# SE2 WRAPPER
# ==============================

def run_se2(subgraph, independent_runs=10, seed=42):
    """
    Run SpeakEasy2 on a subgraph.
    From paper: 10 independent runs, final = highest NMI with all others.
    No resolution parameter needed.
    """
    try:
        membership = se2.cluster(
            subgraph,
            independent_runs=independent_runs,
            seed=seed
        )
        return list(membership)
    except Exception as e:
        print(f"    SE2 error: {e}")
        return None


# ==============================
# RECURSIVE SE2
# ==============================

def recursive_se2(graph, node_indices,
                  max_size=MAX_CLUSTER_SIZE,
                  current_level=0,
                  parent_id="",
                  max_levels=MAX_LEVELS):

    n_nodes = len(node_indices)
    indent  = "  " * current_level

    # ── Leaf conditions ──
    if n_nodes <= max_size or current_level >= max_levels or n_nodes <= 2:
        comm_id = parent_id if parent_id else "0"
        print(f"{indent}Level {current_level}: {n_nodes} nodes → LEAF")
        return {idx: comm_id for idx in node_indices}

    # ── Extract subgraph ──
    subgraph = graph.subgraph(node_indices)

    # ── Handle disconnected subgraph ──
    if not subgraph.is_connected():
        components = subgraph.connected_components()
        print(f"{indent}Level {current_level}: {n_nodes} nodes, {len(components)} components")
        all_assignments = {}
        for comp_idx, component in enumerate(components):
            comp_nodes = [node_indices[i] for i in component]
            comp_id    = f"{parent_id}.{comp_idx}" if parent_id else str(comp_idx)
            all_assignments.update(
                recursive_se2(graph, comp_nodes, max_size,
                              current_level, comp_id, max_levels)
            )
        return all_assignments

    # ── Run SE2 ──
    membership = run_se2(subgraph)

    if membership is None:
        comm_id = parent_id if parent_id else "0"
        print(f"{indent}Level {current_level}: SE2 failed → LEAF")
        return {idx: comm_id for idx in node_indices}

    # Build partition dict
    partition = defaultdict(list)
    for node_idx, comm in enumerate(membership):
        partition[comm].append(node_idx)
    partition = list(partition.values())

    n_communities = len(partition)

    if n_communities == 1:
        comm_id = parent_id if parent_id else "0"
        print(f"{indent}Level {current_level}: {n_nodes} nodes → STOP (1 community)")
        return {node_indices[i]: comm_id for i in range(n_nodes)}

    print(f"{indent}Level {current_level}: SE2 on {n_nodes} nodes → {n_communities} communities")

    # ── Assign + recurse ──
    node_assignments = {}
    for comm_idx, members in enumerate(partition):
        comm_id        = f"{parent_id}.{comm_idx}" if parent_id else str(comm_idx)
        original_nodes = [node_indices[i] for i in members]

        if len(original_nodes) > max_size:
            sub_assignments = recursive_se2(
                graph, original_nodes, max_size,
                current_level + 1, comm_id, max_levels
            )
            node_assignments.update(sub_assignments)
        else:
            for orig_idx in original_nodes:
                node_assignments[orig_idx] = comm_id

    return node_assignments


# ==============================
# RUN
# ==============================

print("\n" + "="*60)
print("RUNNING RECURSIVE SE2")
print("="*60)

hierarchical_assignments = recursive_se2(g, list(range(len(node_list))))

print(f"\n✓ Done: {len(hierarchical_assignments)} nodes assigned")

missing = set(range(len(node_list))) - set(hierarchical_assignments.keys())
print(f"⚠ Missing: {len(missing)}" if missing else "✓ All nodes assigned")

# ==============================
# MAP BACK TO NODE NAMES
# ==============================

hierarchical_named = {
    node_list[idx]: path
    for idx, path in hierarchical_assignments.items()
}

max_depth = max(len(path.split(".")) for path in hierarchical_named.values())
print(f"\nHierarchy depth: {max_depth} levels")

# ==============================
# EXTRACT LEVELS
# ==============================

community_data = {}
for level_idx in range(max_depth):
    level_assignments = {}
    for node, path in hierarchical_named.items():
        parts      = path.split(".")
        level_comm = ".".join(parts[:level_idx + 1])
        level_assignments[node] = level_comm
    community_data[f"level_{level_idx}"] = level_assignments
    print(f"  Level {level_idx}: {len(set(level_assignments.values()))} communities")

# ==============================
# SAVE
# ==============================

statistics = []

for level_idx in range(max_depth):
    level_name    = f"level_{level_idx}"
    node_to_comm  = community_data[level_name]
    comm_to_nodes = defaultdict(list)

    for node, comm in node_to_comm.items():
        comm_to_nodes[comm].append(node)

    sizes = [len(v) for v in comm_to_nodes.values()]

    print(f"\n{level_name.upper()}:")
    print(f"  Communities : {len(comm_to_nodes)}")
    print(f"  Avg size    : {np.mean(sizes):.1f}")
    print(f"  Range       : {np.min(sizes)}-{np.max(sizes)}")

    statistics.append({
        "level"      : level_idx,
        "communities": len(comm_to_nodes),
        "avg_size"   : float(np.mean(sizes)),
        "min_size"   : int(np.min(sizes)),
        "max_size"   : int(np.max(sizes))
    })

    filename = f"{OUTPUT_DIR}/{level_name}_communities.json"
    with open(filename, "w") as f:
        json.dump(dict(comm_to_nodes), f, indent=2)
    print(f"  Saved → {filename}")

with open(f"{OUTPUT_DIR}/hierarchy.json", "w") as f:
    json.dump(hierarchical_named, f, indent=2)

with open(f"{OUTPUT_DIR}/statistics.json", "w") as f:
    json.dump({
        "algorithm"       : "SE2 (SpeakEasy2: Champagne)",
        "paper"           : "Gaiteri et al. Genome Biology 2023",
        "graph_nodes"     : G_nx.number_of_nodes(),
        "graph_edges"     : G_nx.number_of_edges(),
        "max_cluster_size": MAX_CLUSTER_SIZE,
        "hierarchy_depth" : max_depth,
        "levels"          : statistics
    }, f, indent=2)

print("\n" + "="*60)
print("✓ SE2 COMPLETE")
print(f"  Output → {OUTPUT_DIR}/")
print("="*60)