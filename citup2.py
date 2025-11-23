#!/usr/bin/env python
# -*- coding: utf-8 -*-

# CITUP fit-only pipeline that enumerates near-optimal trees (no MACHINA stages).
# NOTE: The ancestry matrix A is removed. Rooted-tree constraints are enforced directly on B; any optional
#       external ancestry (A_input) is handled via B-only constraints.

from gurobipy import *
import argparse, os, sys, time

epsilon = 1e-10

def roundInt(x): return int(x + 0.5)
def isFloat(v):
    try: float(v); return True
    except: return False
def floatToStr(x, d=4): return ("{:." + str(d) + "f}").format(float(x)) if isFloat(x) else x

def safe_val(var):
    """Return variable value even if a later stage failed (uses X or Xn)."""
    for attr in ('X', 'Xn'):
        try:
            return var.getAttr(attr)
        except Exception:
            continue
    return 0.0

def roundVar(var):
    return roundInt(safe_val(var))


# ---------------- CLI ----------------
parser = argparse.ArgumentParser(
    description='CITUP fit-only tree enumeration (no MACHINA migration stages)',
    add_help=True
)

# Required
parser.add_argument('-f','--frequenciesFile', required=True)
parser.add_argument('-o','--pathOutputFilePrefix', required=True)

# Optional ancestry
parser.add_argument('-A','--ancestryMatrixFile', type=str, default=None)
parser.add_argument('--autoAncestryTolerance', type=float, default=None)

# Objective / data options
parser.add_argument('--objective', choices=['mutation','cluster'], default='cluster')
parser.add_argument('--clusterWeight', choices=['size','equal','sqrt'], default='size')
parser.add_argument('--presence-threshold', type=float, default=0.05)
parser.add_argument('--co-presence-reward', type=float, default=0.0)
parser.add_argument('--min-mutations-per-cluster', type=int, default=0)
parser.add_argument('--filter-unbiased-var-ge', type=float, default=None,
                    help='Filter OUT clusters whose perSampleVarSum_unbiased >= threshold (across-sample sum of unbiased within-sample variances).')
parser.add_argument('--fixClusterMapping', action='store_true', default=True)

# Gurobi (global + per-stage)
parser.add_argument('--gurobi-threads', type=int, default=None)
parser.add_argument('--gurobi-max-time', type=float, default=None)
parser.add_argument('--write-mps', type=str, default=None)

# Stage-1 solver controls (fit)
parser.add_argument('--fit-gap', type=float, default=0.0)
parser.add_argument('--fit-gap-abs', type=float, default=0.0)
parser.add_argument('--fit-time-limit', type=float, default=None)

# Enumeration over near-optimal trees
parser.add_argument('--enum-fit-rel', type=float, default=0.0,
                    help='Enumerate trees with fit ≤ best*(1+rel) (or ≤ best+abs).')
parser.add_argument('--enum-fit-abs', type=float, default=0.0,
                    help='Enumerate trees with fit ≤ best+abs (combined with rel).')
parser.add_argument('--enum-max-trees', type=int, default=20,
                    help='Max number of trees to collect (including the best). Set 0 or -1 for unlimited (subject to --enum-time-limit).')
parser.add_argument('--enum-time-limit', type=float, default=None,
                    help='Overall time cap (seconds) for tree enumeration.')
# Number of optimal (fit-minimal) trees to enumerate (top-K by fit).
# When set, overrides near-optimal enumeration and enforces fit == best_fit for all K trees.
parser.add_argument('--numSolutions', dest='num_solutions', type=int, default=None,
                    help='Enumerate exactly K optimal trees (top-K by fit). Overrides enum-max-trees; disables near-optimal expansion (uses fit upper bound = best fit).')

parser.add_argument('--disable-auto-ancestry-caps', action='store_true', default=False,
                    help='Ignore CP-based allowed-edge mask (A_allowed_all) when building B/q/migration variables.')

# Debug
parser.add_argument('--debug-iis', action='store_true', default=False)

# Optional preprocessing of observed frequencies
parser.add_argument('--clip-observed-at', type=float, default=None,
                    help='Clip observed mutation/cluster frequencies at this value before fitting '
                         '(applies to both objectives).')

# Reproduce legacy (buggy) fit behavior across both objectives
parser.add_argument('--legacy-fit', action='store_true', default=False,
                    help='Reproduce legacy mutation fit that used z on all nodes with M=1. '
                         'Applies this residual for BOTH objectives so results match the legacy output.')

# Mutation objective linearization control
parser.add_argument('--mut-abs-bigM', type=float, default=None,
                    help='Override absolute big-M for mutation objective linearization. '
                         'Default: auto = max(1.0, max observed frequency).')

args = parser.parse_args()

def _log_args(parsed):
    """Display all CLI arguments so runs are traceable."""
    print("=== citup7 arguments ===")
    for key in sorted(vars(parsed)):
        print(f"{key} = {getattr(parsed, key)}")
    print("=== end arguments ===")

_log_args(args)

presence_threshold = max(0.0, min(1.0, float(args.presence_threshold)))
co_presence_reward = float(args.co_presence_reward)
min_mut_per_cluster = max(0, int(args.min_mutations_per_cluster))

# ---------------- IO ----------------
f, c = {}, {}
mutIDs, clusterIDs = [], []
clusterID_counts = {}
with open(args.frequenciesFile, "r") as fh:
    hdr = fh.readline().strip().split()
    sampleIDs = [s.strip() for s in hdr[1:-1]]
    numSamples = len(sampleIDs)
    for line in fh:
        cols = line.strip().split()
        assert len(cols) - 2 == numSamples, "Bad row (sample count mismatch): " + line
        mid = cols[0]
        assert mid not in mutIDs, f"Duplicate mutation id: {mid}"
        mutIDs.append(mid)
        f[mid] = {sampleIDs[i]: float(cols[i+1]) for i in range(numSamples)}
        cid = cols[-1]
        if cid not in clusterIDs: clusterIDs.append(cid)
        c[mid] = cid
# Build cluster->mut list and snapshots
cluster_to_muts = {cid: [] for cid in clusterIDs}
for mid in mutIDs:
    cluster_to_muts[c[mid]].append(mid)

# Optional: clip observed frequencies before computing CP/weights, so both objectives see the same targets
if args.clip_observed_at is not None:
    cap = float(args.clip_observed_at)
    for mid in mutIDs:
        for s in sampleIDs:
            if f[mid][s] > cap:
                f[mid][s] = cap

# Big-M for mutation objective linearization: must dominate |f - y| when delta==0.
# y in [0,1]. If user provided --mut-abs-bigM, use it; otherwise auto = max(1.0, max observed f).
try:
    _f_max = max((f[mid][s] for mid in mutIDs for s in sampleIDs), default=1.0)
except Exception:
    _f_max = 1.0
if args.mut_abs_bigM is not None:
    BIGM_MUT_ABS = float(args.mut_abs_bigM)
    if BIGM_MUT_ABS < _f_max - 1e-9:
        print(f"[warn] --mut-abs-bigM={BIGM_MUT_ABS} < max observed frequency {_f_max:.4f}; mutation objective may not match cluster objective.")
else:
    BIGM_MUT_ABS = max(1.0, float(_f_max))

orig_map = dict(c)
orig_mut = list(mutIDs)
orig_clu = list(clusterIDs)

# Apply min-mutations-per-cluster (strict >) first
min_k = max(0, int(args.min_mutations_per_cluster))
kept_min = [cid for cid in orig_clu if len(cluster_to_muts[cid]) > min_k]

# Compute unbiased per-sample variance sum for each original cluster
def _unbiased_sum(cid):
    muts = cluster_to_muts[cid]
    k = len(muts)
    if k <= 1:
        return 0.0
    total_pop = 0.0
    for sname in sampleIDs:
        vals = [f[m][sname] for m in muts]
        mu = (sum(vals)/float(k)) if k else 0.0
        var_pop = (sum((v-mu)**2 for v in vals)/float(k)) if k else 0.0
        total_pop += var_pop
    return total_pop * (float(k)/float(k-1))

unb = {cid: _unbiased_sum(cid) for cid in orig_clu}

# If requested, filter by unbiased variance >= threshold
thr = args.filter_unbiased_var_ge
if thr is not None:
    thr = float(thr)
    retained_clusters = [cid for cid in kept_min if unb[cid] < thr]
else:
    retained_clusters = kept_min

if not retained_clusters:
    raise ValueError('All clusters filtered out by --min-mutations-per-cluster and/or --filter-unbiased-var-ge; consider lowering thresholds.')

# Emit unbiased variance report and concise stdout
report_path = args.pathOutputFilePrefix + '.cluster_variance.tsv'
with open(report_path, 'w') as fv:
    fv.write('clusterID	k	perSampleVarSum_unbiased	threshold	kept\n')
    thr_str = '' if args.filter_unbiased_var_ge is None else str(thr)
    for cid in orig_clu:
        k = len(cluster_to_muts[cid])
        kept = 1 if cid in retained_clusters else 0
        fv.write(str(cid)+'\t'+str(k)+'\t'+('%.6f' % unb[cid])+'\t'+thr_str+'\t'+str(kept)+'\n')
        print('Cluster '+str(cid)+': unbiasedVar=' + ('%.6f' % unb[cid]) + ' kept=' + str(kept))

# Restrict to retained clusters and rebuild counts/mappings
clusterIDs = retained_clusters
mutIDs = [mid for mid in orig_mut if orig_map[mid] in clusterIDs]
cluster_to_muts = {cid: [mid for mid in cluster_to_muts[cid] if mid in mutIDs] for cid in clusterIDs}
clusterID_counts = {cid: len(cluster_to_muts[cid]) for cid in clusterIDs}
c = {mid: orig_map[mid] for mid in mutIDs}

numMutations = len(mutIDs)
numClusters  = len(clusterIDs)
treeSize = numClusters + 1

# cluster prevalences (means) after filtering
CP = {cid: {s: (0.0 if len(cluster_to_muts[cid])==0 else
               sum(f[m][s] for m in cluster_to_muts[cid]) / float(len(cluster_to_muts[cid])))
           for s in sampleIDs}
      for cid in clusterIDs}

omega_cluster = {cid: {s: 1 if CP[cid][s] >= presence_threshold else 0 for s in sampleIDs}
                 for cid in clusterIDs}

# Identify samples that retain at least one tumour cluster after filtering
sample_requires_tumor = {s: any(omega_cluster[cid][s] == 1 for cid in clusterIDs) for s in sampleIDs}

ordered_clusters = list(clusterIDs)
clusterIndex = {cid: i+1 for i,cid in enumerate(ordered_clusters)}  # root=0
# ancestry input
A_input = [[-1]*treeSize for _ in range(treeSize)]
if args.ancestryMatrixFile:
    A_input = []
    with open(args.ancestryMatrixFile,"r") as amf:
        for line in amf:
            line=line.strip()
            if not line: continue
            A_input.append([int(float(x)) for x in line.split()])
    assert len(A_input)==treeSize and all(len(r)==treeSize for r in A_input), "Bad ancestry matrix size"

# ancestry feasibility from CP (optional)
A_allowed_all = [[1]*treeSize for _ in range(treeSize)]
if args.autoAncestryTolerance is not None:
    tol = float(args.autoAncestryTolerance)
    def node_to_cluster(v): return None if v==0 else ordered_clusters[v-1]
    for v in range(treeSize):
        for u in range(treeSize):
            if v==u: A_allowed_all[v][u] = 1; continue
            if u==0: A_allowed_all[v][u] = 0; continue
            if v==0: A_allowed_all[v][u] = 1; continue
            cv, cu = node_to_cluster(v), node_to_cluster(u)
            ok = all(CP[cv][s] + tol >= CP[cu][s] for s in sampleIDs)
            A_allowed_all[v][u] = 1 if ok else 0

# with open(args.pathOutputFilePrefix + ".A_allowed_all_samples.tsv","w") as fa:
#     for v in range(treeSize):
#         fa.write("\t".join(str(A_allowed_all[v][u]) for u in range(treeSize))+"\n")

# Precompute which directed edges are allowed (used to sparsify B/q and migration vars)
allowed_edge = [[False]*treeSize for _ in range(treeSize)]
for v in range(treeSize):
    for u in range(treeSize):
        if v == u:
            allowed_edge[v][u] = False
            continue
        forb = False
        # External ancestry forbids direct edge
        a = A_input[v][u]
        if a in (0, 1) and a == 0:
            forb = True
        # CP-based monotonicity mask forbids direct edge (unless disabled)
        if (not args.disable_auto_ancestry_caps) and (A_allowed_all[v][u] == 0):
            forb = True
        allowed_edge[v][u] = (not forb)

# Convenience list of allowed directed edges (v,u) with v != u
ALLOWED_PAIRS = [(v,u) for v in range(treeSize) for u in range(treeSize)
                 if v != u and allowed_edge[v][u]]

# ---------------- utils ----------------
def setup_env(model, log_suffix=""):
    if args.gurobi_threads is not None:
        try: model.setParam('Threads', max(1,int(args.gurobi_threads)))
        except: pass
    # nodefile/mem (optional via env)
    try:
        nfs = os.environ.get('GRB_NODEFILE_START')
        if nfs: model.setParam('NodefileStart', float(nfs))
    except: pass
    try:
        nfdir = os.environ.get('GRB_NODEFILE_DIR') or os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR')
        if nfdir: model.setParam('NodefileDir', nfdir)
    except: pass
    try:
        mem_mb = os.environ.get('GRB_MEMLIMIT_MB')
        if mem_mb: model.setParam('MemLimit', float(mem_mb))
    except: pass
    if args.gurobi_max_time is not None:
        try: model.setParam('TimeLimit', float(args.gurobi_max_time))
        except: pass
    try: model.setParam('LogFile', args.pathOutputFilePrefix + f"{log_suffix}.gurobi.log")
    except: pass
    # Strengthen presolve by default
    try: model.setParam('Presolve', 2)
    except: pass

# --------------- build core model (fit only) ---------------
def build_core():
    root=0
    m=Model('CITUP_core_fit'); setup_env(m,".core")
    # delta
    delta={}
    for cid in clusterIDs:
        for v in range(treeSize):
            delta[cid,v]=m.addVar(vtype=GRB.BINARY, name=f"delta[{cid},{v}]")
    # each cluster ↦ exactly one node
    for cid in clusterIDs:
        m.addConstr(quicksum(delta[cid,v] for v in range(treeSize))==1)
    # node occupancy: no cluster at root; exactly one per non-root
    for v in range(treeSize):
        if v==root:
            for cid in clusterIDs: m.addConstr(delta[cid, v]==0)
        else:
            m.addConstr(quicksum(delta[cid,v] for cid in clusterIDs)==1)
    # optionally fix mapping
    if args.fixClusterMapping:
        for cid in ordered_clusters:
            vfix = clusterIndex[cid]
            m.addConstr(delta[cid, vfix]==1)
            for v in range(treeSize):
                if v!=vfix: m.addConstr(delta[cid,v]==0)

    # omega[v,s]
    omega={}
    for v in range(treeSize):
        for s in sampleIDs:
            omega[v,s]=m.addVar(vtype=GRB.BINARY, name=f"omega[{v},{s}]")
    # root absent (presence indicator)
    for s in sampleIDs: m.addConstr(omega[root,s]==0)
    # non-root omega = presence of its assigned cluster in that sample
    for v in range(1,treeSize):
        for s in sampleIDs:
            contrib=[delta[cid,v] for cid in clusterIDs if omega_cluster[cid][s]==1]
            if contrib: m.addConstr(omega[v,s]==quicksum(contrib))
            else:       m.addConstr(omega[v,s]==0)
    # no sample ONLY root present: require tumour cluster only when at least one survives filtering
    for s in sampleIDs:
        if sample_requires_tumor.get(s, False):
            m.addConstr(quicksum(omega[v,s] for v in range(1,treeSize)) >= 1, name=f"sample_has_tumor[{s}]")
        else:
            # No surviving cluster has presence in this sample; allow root-only assignment
            m.addConstr(quicksum(omega[v,s] for v in range(1,treeSize)) == 0, name=f"sample_root_only[{s}]")

    # x,y
    x={}
    for v in range(treeSize):
        for s in sampleIDs:
            x[v,s]=m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"x[{v},{s}]")
    for s in sampleIDs:
        m.addConstr(quicksum(x[v,s] for v in range(treeSize))==1.0)

    # B: parent→child (edge) matrix only (A removed) — create only for allowed edges; fix others to 0
    B = {}
    for v in range(treeSize):
        for u in range(treeSize):
            if v == u or not allowed_edge[v][u]:
                B[v,u] = m.addVar(vtype=GRB.BINARY, lb=0, ub=0, name=f"B[{v},{u}]")  # fixed 0
            else:
                B[v,u] = m.addVar(vtype=GRB.BINARY, name=f"B[{v},{u}]")

    # Enforce required ancestries from A_input using only B (path existence via transitive cover on A_input)
    if args.ancestryMatrixFile:
        for v in range(treeSize):
            for u in range(treeSize):
                if v != u and A_input[v][u] == 1:
                    m.addConstr(
                        (B[v,u] if (v,u) in B else 0) + quicksum((B[v,w] if (v,w) in B else 0) * A_input[w][u]
                                                                  for w in range(treeSize) if w != v) >= 1
                    )

    # tree structure on B
    for u in range(treeSize):
        if u == root:
            m.addConstr(quicksum(B[v,u] for v in range(treeSize) if v != u) == 0)
        else:
            m.addConstr(quicksum(B[v,u] for v in range(treeSize) if v != u) == 1)

    # y + linearization only across allowed edges
    y, q = {}, {}
    for v in range(treeSize):
        for s in sampleIDs:
            y[v,s] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"y[{v},{s}]")

    for v, u in ALLOWED_PAIRS:
        for s in sampleIDs:
            q[v,u,s] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"q[{v},{u},{s}]")
            m.addConstr(q[v,u,s] <= y[u,s])
            m.addConstr(q[v,u,s] <= B[v,u])
            m.addConstr(q[v,u,s] >= y[u,s] - (1 - B[v,u]))

    for v in range(treeSize):
        for s in sampleIDs:
            m.addConstr(y[v,s] == x[v,s] + quicksum(q[v2,u,s] for (v2,u) in ALLOWED_PAIRS if v2 == v))

    # Acyclicity via indicator constraints on depths (stronger than big-M)
    depth = {v: m.addVar(vtype=GRB.INTEGER, lb=0, ub=treeSize-1, name=f"depth[{v}]") for v in range(treeSize)}
    m.addConstr(depth[root] == 0)
    for v, u in ALLOWED_PAIRS:
        m.addGenConstrIndicator(B[v,u], True, depth[u] >= depth[v] + 1, name=f"acyc[{v},{u}]")

    # Anti-parallel cut to tighten LP: no 2-cycles
    for v in range(treeSize):
        for u in range(v+1, treeSize):
            m.addConstr(B[v,u] + B[u,v] <= 1)

    # Connectivity: single-commodity flow from root to all nodes (prevents forests and enforces reachability)
    BIG = treeSize - 1
    flow = {}
    for v, u in ALLOWED_PAIRS:
        flow[v,u] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=BIG, name=f"flow[{v},{u}]")
        m.addConstr(flow[v,u] <= BIG * B[v,u])

    for u in range(treeSize):
        inflow  = quicksum(flow[v,u] for v in range(treeSize) if (v,u) in flow)
        outflow = quicksum(flow[u,w] for w in range(treeSize) if (u,w) in flow)
        if u == root:
            m.addConstr(outflow - inflow == treeSize - 1)
        else:
            m.addConstr(inflow - outflow == 1)

    # Redundant but tightening cut: total number of edges is N-1
    m.addConstr(quicksum(B[v,u] for v in range(treeSize) for u in range(treeSize) if v!=u) == treeSize - 1)

    # reward (optional) — only on allowed edges to keep model compact
    eq_presence={}
    match_all={}
    reward_expr=None
    if abs(co_presence_reward)>epsilon and ALLOWED_PAIRS:
        reward_expr = LinExpr()
        for (v,u) in ALLOWED_PAIRS:
            eqs=[]
            for s in sampleIDs:
                eq_presence[v,u,s]=m.addVar(vtype=GRB.BINARY, name=f"eq_presence[{v},{u},{s}]")
                # eq_presence[v,u,s] == 1  iff  omega[v,s] == omega[u,s]
                m.addConstr(eq_presence[v,u,s] <= 1 - omega[v,s] + omega[u,s])
                m.addConstr(eq_presence[v,u,s] <= 1 + omega[v,s] - omega[u,s])
                m.addConstr(eq_presence[v,u,s] >= omega[v,s] + omega[u,s] - 1)
                m.addConstr(eq_presence[v,u,s] >= 1 - omega[v,s] - omega[u,s])
                eqs.append(eq_presence[v,u,s])
            match_all[v,u]=m.addVar(vtype=GRB.BINARY, name=f"match_all[{v},{u}]")
            m.addConstr(match_all[v,u] <= B[v,u])
            for s in sampleIDs:
                m.addConstr(match_all[v,u] <= eq_presence[v,u,s])
            if sampleIDs:
                m.addConstr(match_all[v,u] >= B[v,u] + quicksum(eqs) - len(sampleIDs))
            reward_expr += match_all[v,u]

    # fit objective
    residual_expr=None
    # Legacy bug mode: use the old mutation residual for BOTH objectives, so they match exactly
    if args.legacy_fit:
        residual_expr=QuadExpr()
        z={}
        def _mut_w(cid):
            k=float(clusterID_counts[cid])
            if args.clusterWeight=='equal': return 1.0/k if k>0 else 1.0
            if args.clusterWeight=='sqrt':  return (k**-0.5) if k>0 else 1.0
            return 1.0
        for i,mid in enumerate(mutIDs):
            cid=c[mid]
            for v in range(treeSize):
                for s in sampleIDs:
                    z[mid,v,s]=m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z[{mid},{v},{s}]")
                    # Legacy fit: implicit M=1 on non-selected nodes
                    m.addConstr(z[mid,v,s] >= (delta[cid,v] - 1) + (f[mid][s] - y[v,s]))
                    m.addConstr(z[mid,v,s] >= (delta[cid,v] - 1) - (f[mid][s] - y[v,s]))
                    residual_expr += _mut_w(cid) * z[mid,v,s] * z[mid,v,s]
    elif args.objective=='mutation':
        residual_expr=QuadExpr()
        z={}
        def _mut_w(cid):
            k=float(clusterID_counts[cid])
            if args.clusterWeight=='equal': return 1.0/k if k>0 else 1.0
            if args.clusterWeight=='sqrt':  return (k**-0.5) if k>0 else 1.0
            return 1.0
        for i,mid in enumerate(mutIDs):
            cid=c[mid]
            if args.fixClusterMapping:
                # Mapping is fixed: only the fixed node can host this cluster.
                vfix = clusterIndex[cid]
                for s in sampleIDs:
                    z[mid,vfix,s]=m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z[{mid},{vfix},{s}]")
                    # Exact absolute value without big-M, since delta[cid,vfix]==1
                    m.addConstr(z[mid,vfix,s] >=  (f[mid][s] - y[vfix,s]))
                    m.addConstr(z[mid,vfix,s] >= -(f[mid][s] - y[vfix,s]))
                    residual_expr += _mut_w(cid) * z[mid,vfix,s] * z[mid,vfix,s]
            else:
                for v in range(treeSize):
                    for s in sampleIDs:
                        z[mid,v,s]=m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"z[{mid},{v},{s}]")
                        # Use a safe big-M so when delta==0 these bounds are non-binding (<=0)
                        m.addConstr(z[mid,v,s] >= BIGM_MUT_ABS*(delta[cid,v] - 1) + (f[mid][s] - y[v,s]))
                        m.addConstr(z[mid,v,s] >= BIGM_MUT_ABS*(delta[cid,v] - 1) - (f[mid][s] - y[v,s]))
                        residual_expr += _mut_w(cid) * z[mid,v,s] * z[mid,v,s]
    else:
        w,t={},{}
        for cid in clusterIDs:
            for v in range(treeSize):
                for s in sampleIDs:
                    w[cid,v,s]=m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"w[{cid},{v},{s}]")
                    m.addConstr(w[cid,v,s] <= y[v,s])
                    m.addConstr(w[cid,v,s] <= delta[cid,v])
                    m.addConstr(w[cid,v,s] >= y[v,s] - (1 - delta[cid,v]))
            for s in sampleIDs:
                t[cid,s]=m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"t[{cid},{s}]")
                m.addConstr(t[cid,s] == quicksum(w[cid,v,s] for v in range(treeSize)))
        residual_expr=QuadExpr()
        for cid in clusterIDs:
            k=float(clusterID_counts[cid])
            wgt = 1.0 if args.clusterWeight=='equal' else (k**0.5 if args.clusterWeight=='sqrt' else k)
            for s in sampleIDs:
                residual_expr += wgt * (t[cid,s]-CP[cid][s])*(t[cid,s]-CP[cid][s])

    fit_obj = residual_expr + 0
    if reward_expr is not None:
        fit_obj -= co_presence_reward * reward_expr

    return dict(model=m, root=root, delta=delta, omega=omega, x=x, y=y, B=B,
                fit_obj=fit_obj, residual_expr=residual_expr, reward_expr=reward_expr)

# ----------------- Solve pipeline -----------------
# storage
solutions = []  # collected trees: {core, b_edges, fit, residual_stage1, reward_stage1, stage1_delta, stage1_x, stage1_y}
enumerated_b = []  # all tree structures already enumerated (feasible or not)

best_fit = None
fit_upper = None

# First: solve core once for best fit
core = build_core()
m = core["model"]
m.setObjective(core["fit_obj"], GRB.MINIMIZE)
m.Params.MIPGap    = float(args.fit_gap)
m.Params.MIPGapAbs = float(args.fit_gap_abs)
if args.fit_time_limit is not None: m.Params.TimeLimit = float(args.fit_time_limit)
m.optimize()
if m.Status in (GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD):
    if args.debug_iis:
        try:
            m.computeIIS()
            m.write(args.pathOutputFilePrefix + ".stage1_best.iis.ilp")
            m.write(args.pathOutputFilePrefix + ".stage1_best.lp")
        except: pass
    raise RuntimeError("Stage-1 infeasible/unbounded.")
if m.SolCount==0: raise RuntimeError("Stage-1: no incumbent.")

# Enforce optimality unless explicitly allowed (skip for resource limits)
allow_stage1_subopt = (m.Status in (GRB.TIME_LIMIT, GRB.MEM_LIMIT))
require_opt_stage1 = getattr(args, 'require_optimal_stage1', True)
if m.Status != GRB.OPTIMAL and require_opt_stage1 and not allow_stage1_subopt:
    try:
        bound = m.ObjBound
        val = m.ObjVal
        gap = m.MIPGap
        print(f"[error] Stage-1 not proven optimal (status={m.Status}). Incumbent={val:.6g}, bestBound={bound:.6g}, gap={gap:.3g}", file=sys.stderr)
    except Exception:
        print(f"[error] Stage-1 not proven optimal (status={m.Status}).", file=sys.stderr)
    raise RuntimeError("Stage-1 did not finish optimal and --require-optimal-stage1 is set. Use --allow-suboptimal-stage1 to proceed anyway.")
elif m.Status != GRB.OPTIMAL and allow_stage1_subopt:
    try:
        val = m.ObjVal
        gap = m.MIPGap
        print(f"[warn] Stage-1 ended with status={m.Status}; continuing with incumbent. Incumbent={val:.6g}, gap={gap:.3g}", file=sys.stderr)
    except Exception:
        print(f"[warn] Stage-1 ended with status={m.Status}; continuing with incumbent.", file=sys.stderr)

best_fit = core["fit_obj"].getValue()
# Base enumeration settings (near-optimal by default)
fit_upper = best_fit + max(float(args.enum_fit_abs), float(args.enum_fit_rel)*abs(best_fit))
all_pairs = set(ALLOWED_PAIRS)
enum_deadline = time.time() + float(args.enum_time_limit) if args.enum_time_limit is not None else None
_max_t = int(args.enum_max_trees)
_unlimited = (_max_t <= 0)
target_solutions = float("inf") if _unlimited else _max_t
# K-best mode: enumerate the K best-fit trees (in increasing fit), regardless of ties
kbest_mode = False
if getattr(args, 'num_solutions', None) not in (None, 0):
    _max_t = max(1, int(args.num_solutions))
    _unlimited = False
    target_solutions = _max_t
    kbest_mode = True  # do not cap fit by best_fit; let next solves improve objective to next-best

def capture_B(core):
    B = core['B']
    edges = set()
    for v in range(treeSize):
        for u in range(treeSize):
            if v != u and roundVar(B[v, u]) == 1:
                edges.add((v, u))
    return edges


def capture_stage1_metrics(core):
    fit_val = core["fit_obj"].getValue()
    residual_val = core["residual_expr"].getValue() if core["residual_expr"] is not None else fit_val
    reward_val = co_presence_reward * core["reward_expr"].getValue() if core["reward_expr"] is not None else 0.0
    return fit_val, residual_val, reward_val

def snapshot_stage1_solution(core):
    delta_assign = {}
    for cid in clusterIDs:
        assigned = None
        for v in range(treeSize):
            if roundInt(core["delta"][cid,v].X) == 1:
                assigned = v
                break
        delta_assign[cid] = assigned
    x_vals = {(v,s): core["x"][v,s].X for v in range(treeSize) for s in sampleIDs}
    y_vals = {(v,s): core["y"][v,s].X for v in range(treeSize) for s in sampleIDs}
    return delta_assign, x_vals, y_vals

def compute_root_degree(sol):
    edges = sol.get("b_edges")
    if not edges:
        edges = set()
        B = sol["core"]["B"]
        for v in range(treeSize):
            for u in range(treeSize):
                if v != u and roundVar(B[v, u]) == 1:
                    edges.add((v, u))
    return sum(1 for (p, ch) in edges if p == 0)

nogood_counter = 0
def add_hamming_cut(model, B_vars, edge_set, tag):
    global nogood_counter
    nogood_counter += 1
    model.addConstr(
        quicksum((1 - B_vars[v,u]) for (v,u) in edge_set) +
        quicksum(B_vars[v,u] for (v,u) in (all_pairs - edge_set))
        >= 1,
        name=f"{tag}_{nogood_counter}"
    )

def enumerate_next(prev_edge_sets, label_prefix):
    core_k = build_core()
    mk = core_k["model"]
    # In K-best mode, do NOT cap by fit_upper; otherwise, restrict to near-optimal window
    if not kbest_mode:
        mk.addQConstr(core_k["fit_obj"] <= fit_upper, name="enum_fit_cap")
    for b_prev in prev_edge_sets:
        if not b_prev:
            continue
        add_hamming_cut(mk, core_k["B"], b_prev, label_prefix)
    mk.setObjective(core_k["fit_obj"], GRB.MINIMIZE)
    mk.Params.MIPGap = 0; mk.Params.MIPGapAbs = 0
    if args.fit_time_limit is not None:
        mk.Params.TimeLimit = float(args.fit_time_limit)

    attempts = 0
    while True:
        if enum_deadline is not None and time.time() >= enum_deadline:
            return None
        mk.optimize()
        if mk.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or mk.SolCount == 0:
            return None
        fit_k = core_k["fit_obj"].getValue()
        if (not kbest_mode) and (fit_k > fit_upper + 1e-12):
            return None
        b_edges_k = capture_B(core_k)
        if any(b_edges_k == b_prev for b_prev in prev_edge_sets):
            add_hamming_cut(mk, core_k["B"], b_edges_k, f"{label_prefix}_retry")
            mk.update()
            attempts += 1
            if attempts >= 5:
                return None
            continue
        f0, r0, rw0 = capture_stage1_metrics(core_k)
        delta_snap, x_snap, y_snap = snapshot_stage1_solution(core_k)
        return dict(core=core_k, b_edges=b_edges_k, fit=f0, residual_stage1=r0, reward_stage1=rw0, stage1_delta=delta_snap, stage1_x=x_snap, stage1_y=y_snap)

# Seed enumeration with best-fit tree
f0, r0, rw0 = capture_stage1_metrics(core)
delta_snap, x_snap, y_snap = snapshot_stage1_solution(core)
initial_sol = dict(core=core, b_edges=capture_B(core), fit=f0, residual_stage1=r0, reward_stage1=rw0, stage1_delta=delta_snap, stage1_x=x_snap, stage1_y=y_snap)
enumerated_b.append(initial_sol["b_edges"])
solutions.append(initial_sol)

# Enumerate near-optimal trees until target reached or infeasible
while (_unlimited or len(solutions) < target_solutions):
    if enum_deadline is not None and time.time() >= enum_deadline:
        break
    prev_edge_sets = []
    for edge_set in enumerated_b:
        if edge_set and not any(edge_set == existing for existing in prev_edge_sets):
            prev_edge_sets.append(edge_set)
    next_sol = enumerate_next(prev_edge_sets, "nogood_hamming")
    if next_sol is None:
        break
    enumerated_b.append(next_sol["b_edges"])
    solutions.append(next_sol)
    if (not _unlimited) and len(solutions) >= target_solutions:
        break

if not solutions:
    raise RuntimeError("No feasible trees enumerated.")

for sol in solutions:
    sol["root_degree"] = compute_root_degree(sol)

def _best_selection_key(idx):
    sol = solutions[idx]
    fit_val = sol.get("fit")
    return (
        fit_val if fit_val is not None else float("inf"),
        -sol.get("root_degree", 0),
        idx
    )

best_idx = min(range(len(solutions)), key=_best_selection_key)

# --------------- write outputs ---------------
def write_solution(sol_idx, sol, tag=""):
    core_i = sol["core"]
    suffix = (f"_sol{sol_idx}{tag}" if (sol_idx>0 or tag) else f"{tag}")
    fit_val = sol.get("fit")
    if fit_val is None:
        try:
            fit_val = core_i["fit_obj"].getValue()
        except Exception:
            fit_val = 0.0
    residual_val = sol.get("residual_stage1")
    if residual_val is None:
        try:
            residual_val = core_i["residual_expr"].getValue() if core_i["residual_expr"] is not None else fit_val
        except Exception:
            residual_val = fit_val
    reward_val = sol.get("reward_stage1")
    if reward_val is None:
        try:
            reward_val = co_presence_reward * core_i["reward_expr"].getValue() if core_i["reward_expr"] is not None else 0.0
        except Exception:
            reward_val = 0.0
    b_edges = sorted(sol.get("b_edges", []))

    stage1_delta = sol.get("stage1_delta") or {}
    stage1_x = sol.get("stage1_x") or {}
    stage1_y = sol.get("stage1_y") or {}

    # cluster↦node
    nodeOfCluster = {}
    for cid in clusterIDs:
        node = None
        for v in range(treeSize):
            if roundVar(core_i["delta"][cid, v]) == 1:
                node = v
                break
        nodeOfCluster[cid] = node

    edges = []
    for v in range(treeSize):
        for u in range(treeSize):
            if v != u and roundVar(core_i["B"][v, u]) == 1:
                edges.append((v, u))
    if not edges:
        edges = b_edges

    rootDegree = sum(1 for (p, ch) in edges if p == 0)

    # tree file
    with open(args.pathOutputFilePrefix + f".tree_{sol_idx}{tag}.txt", "w") as fp:
        fp.write("TREE_SCORE\n" + floatToStr(fit_val) + "\n")
        fp.write("TREE_SCORE_NO_REWARD\n" + floatToStr(residual_val) + "\n")
        fp.write("REWARD_CONTRIBUTION\n" + floatToStr(reward_val) + "\n")
        fp.write("ROOT_DEGREE\n" + str(rootDegree) + "\n")
        fp.write("\nPARENT_NODE\tCHILD_NODE\n")
        for (p, ch) in edges:
            fp.write(f"{p}\t{ch}\n")
        fp.write("\nCLUSTER\tNODE_OF_CLUSTER\n")
        for cid in clusterIDs:
            fp.write(f"{cid}\t{nodeOfCluster[cid]}\n")
        fp.write("\nMUTATION\tNODE_OF_MUTATION\n")
        for mid in mutIDs:
            fp.write(f"{mid}\t{nodeOfCluster.get(c[mid])}\n")
        fp.write("\nNODE\tNODE_FREQS\tCLADE_FREQS\n")
        for v in range(treeSize):
            node_vals = [stage1_x.get((v, s), safe_val(core_i["x"][v, s])) for s in sampleIDs]
            clade_vals = [stage1_y.get((v, s), safe_val(core_i["y"][v, s])) for s in sampleIDs]
            fp.write(str(v) + "\t" + ",".join(floatToStr(val + epsilon) for val in node_vals) + "\t")
            fp.write(",".join(floatToStr(val + epsilon) for val in clade_vals) + "\n")
        fp.write("\nMUT_ID\tOBSERVED_FREQS\tINFERRED_FREQS\tABS_OBSERVED_MINUS_INFERRED\n")
        for mid in mutIDs:
            v = nodeOfCluster.get(c[mid])
            fp.write(mid + "\t" + ",".join(floatToStr(f[mid][s]) for s in sampleIDs) + "\t")
            if v is None:
                inferred = [0.0 for _ in sampleIDs]
            else:
                inferred = [stage1_y.get((v, s), safe_val(core_i["y"][v, s])) for s in sampleIDs]
            fp.write(",".join(floatToStr(val) for val in inferred) + "\t")
            fp.write(",".join(floatToStr(abs(f[mid][s] - inferred[idx])) for idx, s in enumerate(sampleIDs)) + "\n")


# write all solutions and mark best-by-fit
with open(args.pathOutputFilePrefix + ".treeScores.tsv","w") as fs:
    fs.write("treeIndex\tfit\trootDegree\tisBestByFit\n")
    for i,sol in enumerate(solutions):
        B = sol["core"]["B"]
        rootDeg = sol.get("root_degree")
        if rootDeg is None:
            rootDeg = sum(1 for u in range(treeSize) if roundVar(B[0,u])==1)
        best_flag = 1 if i==best_idx else 0
        fit_val = sol.get("fit")
        fit_str = "NA" if fit_val is None else floatToStr(fit_val)
        fs.write(f"{i}\t{fit_str}\t{rootDeg}\t{best_flag}\n")

for i,sol in enumerate(solutions):
    write_solution(i, sol, tag="")

# Duplicate best solution with a convenient suffix
# write_solution(best_idx, solutions[best_idx], tag="_best_fit")
