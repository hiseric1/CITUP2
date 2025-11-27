#!/usr/bin/env python3
"""
Simulate spatially mixed bulk samples from a labeled rooted phylogeny while
forcing the resulting cellular prevalences to admit a single valid tree
topology.

Outputs match simulations.py:
  1) mutation-level TSV: mutID, sample0..sample{S-1}, clusterID
  2) tree_topology.txt (parent -> child per line, includes 'root')
  3) tree.newick (rooted Newick; internal names are integer clusterIDs)
  4) tree.png (PNG visualization; Graphviz layout if available, fallback otherwise)
  5) tree_ascii.txt (simple text tree)
  6) clusters.tsv (cluster-level prevalences per sample; rows ordered by clusterID)

Example:
  python simulations3.py --tree-size 5 --num-samples 6 --seed 7 --outdir out_sim
"""
import math
from collections import deque, defaultdict
import argparse
import json
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np

try:
    import networkx as nx
except Exception:
    nx = None
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def random_labeled_rooted_tree(n: int, max_root_out_deg: int = 1) -> Dict[str, str]:
    """Create a labeled rooted tree of size n.
    Start from {'root': None}. Iteratively add labeled children '0','1',...,
    picking a parent from the deque 'vertices' with a bias that favors the last ~37.5% of entries.
    Limit the out-degree of the special 'root' node to max_root_out_deg.
    """
    if n < 1:
        raise ValueError("tree size n must be >= 1")
    tree: Dict[str, str] = {'root': None}
    root_out_deg = 0
    vertices = deque(['root'])

    for v in range(n - 1):
        # biased sampling to choose parent
        num_vertices = len(vertices)
        higher_prob_vertices = math.floor(num_vertices * 0.375)

        if higher_prob_vertices == 0:
            p = [1.0 / num_vertices] * num_vertices
        else:
            lower_prob_vertices = num_vertices - higher_prob_vertices
            p = ([0.4 / lower_prob_vertices] * lower_prob_vertices +
                 [0.6 / higher_prob_vertices] * higher_prob_vertices)

        parent = np.random.choice(list(vertices), p=p)
        child = str(v)
        tree[child] = parent
        vertices.append(child)

        if parent == 'root':
            root_out_deg += 1
            if root_out_deg == max_root_out_deg:
                # prevent adding more children to 'root'
                # by removing it from the candidate deque
                vertices.popleft()

    return tree


def get_vertices(tree: Dict[str, str]) -> List[str]:
    # Exclude the special 'root' entry, preserve insertion order (like the notebook)
    verts = list(tree.keys())
    return verts[1:]


def get_all_descendants_inclusive(child_to_parent: Dict[str, str]) -> Dict[str, set]:
    """Return {node: set of node and all descendants} for every non-root node."""
    parent_to_children: Dict[str, List[str]] = defaultdict(list)
    nodes: List[str] = []
    for child, parent in child_to_parent.items():
        if child == "root":
            continue
        nodes.append(child)
        if parent is not None:
            parent_to_children[parent].append(child)

    descendants: Dict[str, set] = {}

    def dfs(node: str) -> set:
        if node in descendants:
            return descendants[node]
        desc = {node}
        for ch in parent_to_children.get(node, []):
            desc |= dfs(ch)
        descendants[node] = desc
        return desc

    for node in nodes:
        if node not in descendants:
            dfs(node)
    return descendants


def build_tree_metadata(tree: Dict[str, str], vertices: List[str]) -> Dict[str, dict]:
    """Pre-compute helpers: children, parents, depths, coordinates, leaves, ancestors."""
    metadata: Dict[str, dict] = {}
    if not vertices:
        metadata["leaves"] = []
        metadata["ancestors"] = {}
        metadata["depth"] = {}
        metadata["coords"] = {}
        metadata["paths"] = {}
        metadata["node_to_idx"] = {}
        return metadata

    node_to_idx = {v: i for i, v in enumerate(vertices)}
    parent_map: Dict[str, str] = {}
    kids: Dict[str, List[str]] = defaultdict(list)
    root_children: List[str] = []
    for child, parent in tree.items():
        if parent is None:
            continue
        if parent == "root":
            if child in node_to_idx:
                root_children.append(child)
        else:
            kids[parent].append(child)
            parent_map[child] = parent

    for lst in kids.values():
        lst.sort(key=lambda x: node_to_idx.get(x, sys.maxsize))
    root_children.sort(key=lambda x: node_to_idx.get(x, sys.maxsize))
    if not root_children:
        root_children = [vertices[0]]

    depth: Dict[str, int] = {}

    def dfs_depth(node: str, d: int):
        depth[node] = d
        for ch in kids.get(node, []):
            dfs_depth(ch, d + 1)

    for rc in root_children:
        dfs_depth(rc, 0)
    for v in vertices:
        depth.setdefault(v, 0)

    leaves = [v for v in vertices if len(kids.get(v, [])) == 0]
    if not leaves:
        leaves = vertices.copy()

    # Assign horizontal positions by leaf order, then center internal nodes.
    x_pos: Dict[str, float] = {}
    curx = 0.0

    def assign_leaves(node: str):
        nonlocal curx
        chs = kids.get(node, [])
        if not chs:
            if node not in x_pos:
                x_pos[node] = curx
                curx += 1.0
        else:
            for ch in chs:
                assign_leaves(ch)

    for rc in root_children:
        assign_leaves(rc)
    if not x_pos:
        for v in vertices:
            x_pos[v] = float(node_to_idx[v])

    def assign_internal(node: str) -> float:
        chs = kids.get(node, [])
        if not chs:
            return x_pos.get(node, 0.0)
        xs = [assign_internal(c) for c in chs]
        x = sum(xs) / len(xs)
        x_pos[node] = x
        return x

    for rc in root_children:
        assign_internal(rc)

    coords = {v: np.array([x_pos.get(v, float(node_to_idx[v])), float(depth[v])]) for v in vertices}

    ancestors: Dict[str, set] = {}
    paths: Dict[str, set] = {}
    for v in vertices:
        anc = set()
        cur = parent_map.get(v)
        while cur is not None and cur != "root":
            anc.add(cur)
            cur = parent_map.get(cur)
        ancestors[v] = anc
        paths[v] = anc | {v}

    metadata["node_to_idx"] = node_to_idx
    metadata["parent"] = parent_map
    metadata["children"] = kids
    metadata["root_children"] = root_children
    metadata["depth"] = depth
    metadata["coords"] = coords
    metadata["leaves"] = leaves
    metadata["ancestors"] = ancestors
    metadata["paths"] = paths
    return metadata


def generate_spatial_usage_matrix(
    vertices: List[str],
    metadata: Dict[str, dict],
    num_samples: int,
    sample_prefix: str = "sample",
    min_self_usage: float = 0.01,
    spatial_spread: float = 0.35,
) -> Tuple[np.ndarray, List[str]]:
    """Simulate spatial biopsies with Gaussian falloff so every entry is > 0."""
    L = len(vertices)
    num_samples = max(1, int(num_samples))
    colnames = [f"{sample_prefix}{i}" for i in range(num_samples)]
    if L == 0:
        return np.zeros((0, num_samples), dtype=float), colnames

    coords: Dict[str, np.ndarray] = metadata.get("coords", {})
    depth_map: Dict[str, int] = metadata.get("depth", {})
    leaves: List[str] = metadata.get("leaves", []) or vertices
    paths: Dict[str, set] = metadata.get("paths", {})

    xs = [pos[0] for pos in coords.values()] or [0.0]
    width = max(xs) - min(xs) if len(xs) >= 2 else 1.0
    sigma_base = max(spatial_spread, 0.1)
    sigma_min = sigma_base * (0.6 + 0.4 * width / max(width, 1.0))
    sigma_max = sigma_min * 1.8
    min_self_usage = max(1e-4, float(min_self_usage))

    values = np.zeros((L, num_samples), dtype=float)
    for s in range(num_samples):
        focus_leaf = leaves[s % len(leaves)]
        center = coords.get(focus_leaf, np.zeros(2)) + np.random.normal(scale=sigma_min * 0.25, size=2)
        if len(leaves) > 1:
            neighbour_candidates = [lf for lf in leaves if lf != focus_leaf]
            if neighbour_candidates:
                neighbour = np.random.choice(neighbour_candidates)
                mix = np.random.uniform(0.15, 0.45)
                center = (1.0 - mix) * center + mix * (coords.get(neighbour, np.zeros(2)) +
                                                      np.random.normal(scale=sigma_min * 0.2, size=2))
        sigma = np.random.uniform(sigma_min, sigma_max)
        depth_decay = np.random.uniform(0.65, 0.9)
        branch_bonus = np.random.uniform(1.25, 2.3)

        for idx, node in enumerate(vertices):
            position = coords.get(node, np.zeros(2))
            dist = np.linalg.norm(position - center)
            spatial_weight = math.exp(-(dist ** 2) / (2.0 * sigma ** 2))
            depth_factor = depth_decay ** depth_map.get(node, 0)
            bonus = 1.0
            if node in paths.get(focus_leaf, {focus_leaf}):
                bonus += branch_bonus
            weight = min_self_usage + spatial_weight * depth_factor * bonus
            weight += np.random.uniform(0.0, min_self_usage * 0.25)
            values[idx, s] = weight

        values[:, s] = np.maximum(values[:, s], min_self_usage)
        col_sum = values[:, s].sum()
        if col_sum > 0:
            values[:, s] /= col_sum

    return values, colnames


def enforce_unique_parent_candidates(
    usage: np.ndarray,
    vertices: List[str],
    metadata: Dict[str, dict],
    C: np.ndarray,
    margin: float = 0.02,
    tol: float = 1e-4,
    max_iter: int = 8,
) -> np.ndarray:
    """Amplify per-cluster signals so non-ancestors become infeasible parents."""
    if usage.size == 0 or not vertices:
        return usage
    ancestors = metadata.get("ancestors", {})
    L, S = usage.shape
    margin = max(margin, 1e-4)

    for _ in range(max_iter):
        prevalence = C @ usage
        changed = False
        for child_idx, child in enumerate(vertices):
            child_vals = prevalence[child_idx, :]
            ancestor_nodes = ancestors.get(child, set())
            for cand_idx, cand in enumerate(vertices):
                if cand == child or cand in ancestor_nodes:
                    continue
                diff = prevalence[cand_idx, :] - child_vals
                if np.all(diff >= -tol):
                    sample = int(np.argmin(diff))
                    delta = diff[sample] + margin
                    if delta <= 0:
                        delta = margin
                    usage[child_idx, sample] += delta
                    changed = True
        if not changed:
            break
        usage = np.maximum(usage, tol)
        col_sums = usage.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        usage = usage / col_sums
    return usage


def clonal_matrix_from_tree(vertices: List[str], descendants_inclusive: Dict[str, set]) -> np.ndarray:
    L = len(vertices)
    C = np.zeros((L, L), dtype=int)
    for i, v in enumerate(vertices):
        desc = descendants_inclusive.get(v, {v})
        mask = np.isin(vertices, list(desc))
        C[i, :] = mask.astype(int)
    return C


def simulate_mutation_read_data(
    prevalence: np.ndarray,
    colnames: List[str],
    min_mutations: int,
    max_mutations: int,
    coverage_mean: float,
    coverage_shape: float,
    vaf_heterozygosity: float,
) -> Tuple[List[dict], List[dict]]:
    """Generate per-mutation VAFs plus read counts."""
    L, S = prevalence.shape
    mutation_rows: List[dict] = []
    read_rows: List[dict] = []
    if L == 0:
        return mutation_rows, read_rows

    mut_min = max(1, int(min_mutations))
    mut_max = max(mut_min, int(max_mutations))
    mean_cov = max(1.0, float(coverage_mean))
    shape = max(1e-3, float(coverage_shape))
    scale = mean_cov / shape
    vaf_scale = max(0.0, float(vaf_heterozygosity))

    mut_idx = 0
    for cid in range(L):
        num_mut = np.random.randint(mut_min, mut_max + 1)
        for _ in range(num_mut):
            coverage = np.random.gamma(shape, scale, size=S)
            coverage = np.maximum(1, np.round(coverage).astype(int))
            dcf = prevalence[cid, :]
            target_vaf = np.clip(dcf * vaf_scale, 0.0, 1.0)
            alt = np.random.binomial(coverage, target_vaf)
            obs_vaf = np.divide(alt, coverage, out=np.zeros_like(alt, dtype=float), where=coverage > 0)
            mut_id = f"mut{mut_idx}"
            mut_idx += 1
            mutation_rows.append(
                {
                    "mut_id": mut_id,
                    "cluster_id": str(cid),
                    "values": obs_vaf.tolist(),
                }
            )
            read_rows.append(
                {
                    "mut_id": mut_id,
                    "cluster_id": str(cid),
                    "alt_counts": alt.tolist(),
                    "total_counts": coverage.tolist(),
                }
            )
    return mutation_rows, read_rows


def write_mutations_tsv(
    path: Path,
    prevalence: np.ndarray,
    colnames: List[str],
    mutations_per_cluster: int = 1,
    mutation_rows: Optional[List[dict]] = None,
):
    """Emit mutation-level table.
    Columns: mutID, sample0.., clusterID
    clusterID = row index of the cluster in 'vertices' order.

    - If mutation_rows is provided (simulate-reads mode), write those rows verbatim.
    - Otherwise behaves like the classic simulator using --mutations-per-cluster.
    """
    L, S = prevalence.shape
    with path.open("w") as f:
        header = ["mutID"] + colnames + ["clusterID"]
        f.write("\t".join(header) + "\n")

        if mutation_rows is not None:
            for row in mutation_rows:
                vals = [f"{val:.6g}" for val in row["values"]]
                f.write("\t".join([row["mut_id"]] + vals + [row["cluster_id"]]) + "\n")
            return

        m = max(1, int(mutations_per_cluster))
        if m == 1:
            for cid in range(L):
                mut_id = f"mut{cid}"
                vals = [f"{prevalence[cid, j]:.6g}" for j in range(S)]
                f.write("\t".join([mut_id] + vals + [str(cid)]) + "\n")
        else:
            next_mut_idx = 0
            for cid in range(L):
                vals = [f"{prevalence[cid, j]:.6g}" for j in range(S)]
                for _ in range(m):
                    mut_id = f"mut{next_mut_idx}"
                    next_mut_idx += 1
                    f.write("\t".join([mut_id] + vals + [str(cid)]) + "\n")


def write_mutation_read_counts(path: Path, read_rows: List[dict], colnames: List[str]):
    """Write alt/total read counts per mutation."""
    if not read_rows:
        return
    header = ["mutID", "clusterID"]
    for col in colnames:
        header.extend([f"{col}_alt", f"{col}_total"])
    with path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for row in read_rows:
            parts = [row["mut_id"], row["cluster_id"]]
            for idx in range(len(colnames)):
                parts.append(str(row["alt_counts"][idx]))
                parts.append(str(row["total_counts"][idx]))
            f.write("\t".join(parts) + "\n")


def write_cluster_averages_from_mutations(
    path: Path,
    mutation_rows: List[dict],
    colnames: List[str],
    num_clusters: int,
):
    """Derive cluster-level prevalences by averaging mutation VAFs."""
    if not mutation_rows:
        return
    L = num_clusters
    S = len(colnames)
    sums = np.zeros((L, S), dtype=float)
    counts = np.zeros(L, dtype=int)
    for row in mutation_rows:
        cid = int(row["cluster_id"])
        vals = np.array(row["values"], dtype=float)
        sums[cid, :] += vals
        counts[cid] += 1
    with path.open("w") as f:
        header = ["clusterID"] + colnames + ["numMutations"]
        f.write("\t".join(header) + "\n")
        for cid in range(L):
            if counts[cid] > 0:
                vals = sums[cid, :] / counts[cid]
            else:
                vals = np.zeros(S, dtype=float)
            vals = np.clip(vals * 2.0, 0.0, 1.0)
            formatted = [f"{val:.6g}" for val in vals]
            f.write("\t".join([str(cid)] + formatted + [str(counts[cid])]) + "\n")


def write_clusters_tsv(path: Path, prevalence: np.ndarray, colnames: List[str]):
    """Cluster-level prevalences per sample. Rows align with clusterID order."""
    L, S = prevalence.shape
    with path.open("w") as f:
        header = ["clusterID"] + colnames
        f.write("\t".join(header) + "\n")
        for cid in range(L):
            vals = [f"{prevalence[cid, j]:.6g}" for j in range(S)]
            f.write("\t".join([str(cid)] + vals) + "\n")


def write_tree_topology(path: Path, tree: Dict[str, str]):
    with path.open("w") as f:
        for child, parent in tree.items():
            if parent is None:
                continue
            f.write(f"{parent} -> {child}\n")


def tree_to_newick(tree: Dict[str, str], vertices: List[str]) -> str:
    """Produce Newick using integer clusterIDs as node labels.
    We treat the effective root as the (last) node whose parent == 'root'.
    """
    # Build parent->children
    kids: Dict[str, List[str]] = defaultdict(list)
    effective_root = None
    for child, parent in tree.items():
        if parent == 'root':
            effective_root = child
        elif parent is not None:
            kids[parent].append(child)
    if effective_root is None:
        effective_root = 'root'

    # Map vertex label to clusterID (order in 'vertices')
    vid2cid = {v: i for i, v in enumerate(vertices)}

    def rec(node: str) -> str:
        chs = kids.get(node, [])
        if not chs:
            return str(vid2cid.get(node, node))  # leaves
        inner = ",".join(rec(c) for c in chs)
        return f"({inner}){vid2cid.get(node, node)}"

    return rec(effective_root) + ";"


def draw_tree_png(path: Path, tree: Dict[str, str], vertices: List[str], title: str = "Phylogeny"):
    """Draw a top-to-bottom phylogeny with leaves aligned at the bottom.
    - Excludes the literal 'root' from the drawing; starts from the first real cluster under 'root'.
    - Leaves are evenly spaced horizontally and share the same bottom y-level.
    - Internal nodes are centered above their descendants.
    """
    if nx is None or plt is None:
        return

    # Build children map and find the effective root (last node whose parent == 'root')
    kids: Dict[str, List[str]] = defaultdict(list)
    effective_root = None
    for c, p in tree.items():
        if p == 'root':
            effective_root = c
        elif p is not None:
            kids[p].append(c)
    if effective_root is None:
        # Nothing meaningful to draw
        return

    # Collect subtree nodes reachable from effective_root (exclude literal 'root')
    subtree_nodes: List[str] = []
    def dfs_collect(u: str):
        subtree_nodes.append(u)
        for v in kids.get(u, []):
            dfs_collect(v)
    dfs_collect(effective_root)

    # Build DiGraph for this subtree
    G = nx.DiGraph()
    for u in subtree_nodes:
        G.add_node(u)
        for v in kids.get(u, []):
            G.add_edge(u, v)

    # Map vertex label to cluster IDs for labeling
    vid2cid = {v: i for i, v in enumerate(vertices)}
    labels = {n: f"CID {vid2cid[n]}" for n in G.nodes}

    # Compute depths from effective_root
    depth: Dict[str, int] = {}
    def dfs_depth(u: str, d: int):
        depth[u] = d
        for v in kids.get(u, []):
            dfs_depth(v, d + 1)
    dfs_depth(effective_root, 0)

    # Identify leaves in this subtree
    leaves = [n for n in subtree_nodes if len(kids.get(n, [])) == 0]
    # Deterministic left-to-right order by clusterID
    leaves.sort(key=lambda x: vid2cid[x])

    # Assign x positions to leaves, then set internal nodes to the mean of their children's x
    x_pos: Dict[str, float] = {}
    curx = 0.0
    leaf_spacing = 1.0
    for leaf in leaves:
        x_pos[leaf] = curx
        curx += leaf_spacing

    def assign_internal_x(u: str) -> float:
        ch = kids.get(u, [])
        if not ch:
            return x_pos[u]
        xs = [assign_internal_x(v) for v in ch]
        x = sum(xs) / len(xs)
        x_pos[u] = x
        return x
    assign_internal_x(effective_root)

    # Strict layered layout: y by depth so all nodes at the same depth align horizontally
    level_spacing = 1.5
    pos = {n: (x_pos[n], depth[n] * level_spacing) for n in subtree_nodes}

    # Figure size scaled to leaves and depth
    max_depth = max(depth.values()) if depth else 0
    fig_w = max(6, 1 + len(leaves) * 1.0)
    fig_h = max(4, 1 + (max_depth + 1) * 1.2)

    plt.figure(figsize=(fig_w, fig_h))
    nx.draw(G, pos, with_labels=False, node_size=900, arrows=True, arrowsize=12)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
    plt.title(title)
    plt.axis("off")
    # Top-to-bottom: invert y so root is at the top and leaves at the bottom
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_tree_ascii(path: Path, tree: Dict[str, str]):
    # Build parent->children and find effective root
    kids: Dict[str, List[str]] = defaultdict(list)
    effective_root = None
    for c, p in tree.items():
        if p == 'root':
            effective_root = c
        elif p is not None:
            kids[p].append(c)
    if effective_root is None:
        effective_root = 'root'

    lines: List[str] = []

    def rec(node: str, prefix: str = "", is_last: bool = True):
        connector = "└─ " if is_last else "├─ "
        if node == "root":
            lines.append("root")
        else:
            lines.append(prefix + connector + node)
        chs = kids.get(node, [])
        for i, ch in enumerate(chs):
            last = (i == len(chs) - 1)
            rec(ch, prefix + ("   " if is_last else "│  "), last)

    rec(effective_root, "", True)
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Simulate tree and outputs (mutation TSV, tree files) following the notebook's logic.")
    ap.add_argument("--tree-size", type=int, default=5, help="Total size INCLUDING 'root' (default: 5).")
    ap.add_argument("--num-samples", type=int, default=6, help="Number of samples/columns (default: 6).")
    ap.add_argument("--seed", type=int, default=7, help="Random seed for NumPy RNG (default: 7).")
    ap.add_argument("--max-root-out-deg", type=int, default=1, help="Max out-degree of special 'root' (default: 1).")
    ap.add_argument("--outdir", type=str, default="out_sim", help="Output directory (default: out_sim).")
    ap.add_argument("--run-name", type=str, default="", help="Optional prefix for output filenames, e.g., 'N5'.")
    ap.add_argument(
        "--mutations-per-cluster", "-m",
        type=int,
        default=1,
        help=(
            "Number of mutations to emit per cluster (default: 1). "
            "Tree topology and cluster prevalences remain unchanged; rows are duplicated per cluster."
        ),
    )
    ap.add_argument(
        "--simulate-reads",
        action="store_true",
        help=(
            "Generate per-mutation read counts using binomial sampling from descendant cell "
            "fractions. When set, --mutations-per-cluster is ignored and new rows are drawn "
            "randomly between --mutations-per-cluster-min and --mutations-per-cluster-max."
        ),
    )
    ap.add_argument(
        "--mutations-per-cluster-min",
        type=int,
        default=10,
        help="Minimum number of mutations to assign per cluster when --simulate-reads is enabled (default: 10).",
    )
    ap.add_argument(
        "--mutations-per-cluster-max",
        type=int,
        default=50,
        help="Maximum number of mutations to assign per cluster when --simulate-reads is enabled (default: 50).",
    )
    ap.add_argument(
        "--mean-coverage",
        type=float,
        default=120.0,
        help="Mean sequencing coverage per mutation used when --simulate-reads is enabled (default: 120).",
    )
    ap.add_argument(
        "--coverage-shape",
        type=float,
        default=5.0,
        help="Gamma shape parameter controlling coverage variability when --simulate-reads is set (default: 5).",
    )
    ap.add_argument(
        "--vaf-heterozygosity",
        type=float,
        default=0.5,
        help=(
            "Scale factor applied to descendant cell fractions to obtain expected VAF during read simulation. "
            "Default 0.5 corresponds to heterozygous mutations."
        ),
    )
    ap.add_argument(
        "--freq-noise-percent",
        type=float,
        default=0.0,
        help=(
            "Per-cell multiplicative noise: for each mutation x sample value v, draw delta in [-P, P]% "
            "uniformly and set v -> v * (1 + delta). Example: 5 means delta ~ U(-0.05, 0.05) per cell. "
            "Default: 0 (no noise)."
        ),
    )
    ap.add_argument(
        "--corrupt-freq-percent",
        type=float,
        default=0.0,
        help=(
            "Randomly replace Q% of all mutation x sample frequency values with independent "
            "U(0,1) draws. Applies after --freq-noise-percent. Default: 0 (no corruption)."
        ),
    )
    ap.add_argument(
        "--min-self-usage",
        type=float,
        default=0.01,
        help="Minimum baseline usage each cluster keeps in every sample (avoids zeros; default 0.01).",
    )
    ap.add_argument(
        "--spatial-spread",
        type=float,
        default=0.35,
        help="Controls the spatial Gaussian falloff (larger -> flatter biopsies). Default 0.35.",
    )
    ap.add_argument(
        "--unique-parent-margin",
        type=float,
        default=0.02,
        help="Margin added when amplifying child signals so that only true ancestors remain feasible parents.",
    )
    ap.add_argument(
        "--unique-parent-iters",
        type=int,
        default=8,
        help="Max iterations spent enforcing unique parent candidates (default: 8).",
    )
    args = ap.parse_args()

    # Seed global NumPy RNG like the notebook
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prefix = (args.run_name + "__") if args.run_name else ""

    # Build tree and vertex list
    tree = random_labeled_rooted_tree(args.tree_size, max_root_out_deg=args.max_root_out_deg)
    vertices = get_vertices(tree)  # these are the clusters; clusterID = index in this list
    L = len(vertices)

    metadata = build_tree_metadata(tree, vertices)
    usage_values, colnames = generate_spatial_usage_matrix(
        vertices,
        metadata,
        args.num_samples,
        sample_prefix="sample",
        min_self_usage=args.min_self_usage,
        spatial_spread=args.spatial_spread,
    )
    descendants = get_all_descendants_inclusive(tree)
    C = clonal_matrix_from_tree(vertices, descendants)
    usage_values = enforce_unique_parent_candidates(
        usage_values,
        vertices,
        metadata,
        C,
        margin=args.unique_parent_margin,
        tol=max(1e-4, args.unique_parent_margin * 0.25),
        max_iter=max(1, int(args.unique_parent_iters)),
    )
    prevalence = C @ usage_values  # (L x L) @ (L x S) -> (L x S)

    # Optional per-cell ±P% multiplicative noise (uniform in [-P, P] per cell)
    if args.freq_noise_percent and args.freq_noise_percent != 0.0:
        p = abs(float(args.freq_noise_percent)) / 100.0
        deltas = np.random.uniform(-p, p, size=prevalence.shape)
        prevalence = prevalence * (1.0 + deltas)
        # Keep within [0, 1]
        prevalence = np.clip(prevalence, 0.0, 1.0)

    # Optionally corrupt a fraction of cells to completely random values in [0,1]
    if args.corrupt_freq_percent and args.corrupt_freq_percent != 0.0:
        q = max(0.0, min(100.0, float(args.corrupt_freq_percent))) / 100.0
        Lc, Sc = prevalence.shape
        total_cells = Lc * Sc
        num_corrupt = int(round(q * total_cells))
        if num_corrupt > 0:
            flat_idx = np.random.choice(total_cells, size=num_corrupt, replace=False)
            random_vals = np.random.uniform(0.0, 1.0, size=num_corrupt)
            flat_view = prevalence.ravel()
            flat_view[flat_idx] = random_vals
            # No need to clip; values already in [0,1]

    mutation_rows: Optional[List[dict]] = None
    mutation_read_rows: Optional[List[dict]] = None
    if args.simulate_reads:
        mutation_rows, mutation_read_rows = simulate_mutation_read_data(
            prevalence,
            colnames,
            args.mutations_per_cluster_min,
            args.mutations_per_cluster_max,
            args.mean_coverage,
            args.coverage_shape,
            args.vaf_heterozygosity,
        )

    # Write outputs
    write_mutations_tsv(
        outdir / f"{prefix}mutations.tsv",
        prevalence,
        colnames,
        mutations_per_cluster=args.mutations_per_cluster,
        mutation_rows=mutation_rows,
    )
    write_clusters_tsv(outdir / f"{prefix}clusters.tsv", prevalence, colnames)
    write_tree_topology(outdir / f"{prefix}tree_topology.txt", tree)
    (outdir / f"{prefix}tree.json").write_text(json.dumps(tree, indent=2) + "\n")
    (outdir / f"{prefix}tree.newick").write_text(tree_to_newick(tree, vertices) + "\n")
    write_tree_ascii(outdir / f"{prefix}tree_ascii.txt", tree)
    draw_tree_png(outdir / f"{prefix}tree.png", tree, vertices, title=f"Phylogeny (clusters={L}, samples={args.num_samples})")
    if mutation_read_rows:
        write_mutation_read_counts(outdir / f"{prefix}mutation_read_counts.tsv", mutation_read_rows, colnames)
        write_cluster_averages_from_mutations(
            outdir / f"{prefix}clusters_from_mutations.tsv",
            mutation_rows,
            colnames,
            L,
        )

    # Minimal console output
    print(f"Tree size (incl. 'root'): {args.tree_size}  -> clusters L = {L}")
    print(f"Wrote outputs in: {outdir.resolve()}")
    print("  - mutations.tsv")
    print("  - clusters.tsv")
    print("  - tree_topology.txt")
    print("  - tree.json")
    print("  - tree.newick")
    print("  - tree_ascii.txt")
    print("  - tree.png")
    if mutation_read_rows:
        print("  - mutation_read_counts.tsv")
        print("  - clusters_from_mutations.tsv")
    print(f"Samples emitted: {args.num_samples}")
    if args.freq_noise_percent and args.freq_noise_percent != 0.0:
        print(f"Applied per-cell uniform ±{abs(args.freq_noise_percent)}% multiplicative noise")
    if args.corrupt_freq_percent and args.corrupt_freq_percent != 0.0:
        print(f"Corrupted {abs(args.corrupt_freq_percent)}% of values to U(0,1) random")


if __name__ == "__main__":
    main()
