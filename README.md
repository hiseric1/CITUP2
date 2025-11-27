# CITUP2

CITUP2 is a integrative combinatorial optimization framework that reconstructs clonal trees from descendant cell fractions (DCFs) of mutational clusters. Python script (`citup2.py`) builds a mixed-integer model with Gurobi to fit clonal trees to per-mutation cellular prevalence measurements. It enforces rooted-tree structure directly on the parent–child matrix, can honor an optional ancestry mask, and enumerates top k optimal trees that explain the input clusters.

## Requirements
- Python with `gurobipy` installed and a valid Gurobi license available in the environment (install via `pip install -r requirements.txt` once your Gurobi license is configured).
- Input frequencies file in the tab-separated format below; output directories referenced by `--pathOutputFilePrefix` must already exist.

## Input Format
Tab-separated table with one mutation per row, all sample columns in the middle, and the cluster label in the final column:

```
mutID	sample0	sample1	clusterID
M1	0.40	0.30	clusterA
M2	0.18	0.52	clusterB
```

Mutations must be unique. Cluster names can be any string. Clusters with `<= --min-mutations-per-cluster` mutations (strictly greater than the threshold are kept) are dropped before solving; you can also exclude high-variance clusters via `--filter-unbiased-var-ge`.

## Typical Run
```bash
python citup2.py \
  --frequenciesFile data/clusters.tsv \
  --pathOutputFilePrefix results/run1/cn_fit \
  --objective cluster \
```
Outputs will be written with the given prefix (e.g., `results/run1/cn_fit.tree_0.txt`).

## Key Options
- **Required**: `-f/--frequenciesFile`, `-o/--pathOutputFilePrefix`.
- **Objectives**: `--objective {mutation,cluster}` (default `cluster`), `--clusterWeight {size,equal,sqrt}`, `--presence-threshold`, optional `--co-presence-reward` for rewarding parent/child pairs present in the same samples.
- **Ancestry guidance**: `--ancestryMatrixFile` to force/forbid specific ancestries (entries 1/0/-1), `--autoAncestryTolerance` to mask disallowed edges based on cellular prevalences, `--fixClusterMapping` keeps the input cluster order tied to node indices (default on).
- **Data filtering**: `--min-mutations-per-cluster`, `--filter-unbiased-var-ge` to drop clusters with large within-sample variance, `--clip-observed-at` to cap frequencies before fitting.
- **Optimization limits**: `--fit-gap`, `--fit-gap-abs`, `--fit-time-limit`, `--gurobi-max-time`, `--gurobi-threads`, `--write-mps`, `--debug-iis`.
- **Enumeration**: `--numSolutions K` enumerates the K best-fit trees exactly. 

## Outputs
- `<prefix>.cluster_variance.tsv`: per-cluster counts and unbiased variance report after filtering.
- `<prefix>.gurobi.log`: solver log mirrored from Gurobi.
- `<prefix>.treeScores.tsv`: fit value and root degree for each enumerated tree; best-by-fit is marked.
- `<prefix>.tree_{k}.txt`: tree structure, cluster/mutation assignments, node/clade frequencies, and residuals for each solution.
