# CITUP2

CITUP2 is a integrative combinatorial optimization framework that reconstructs clonal trees from descendant cell fractions (DCFs) of mutational clusters. Python script (`citup2.py`) builds a mixed-integer model with Gurobi to fit clonal trees to per-mutation cellular prevalence measurements. It enforces rooted-tree structure directly on the parent–child matrix, and enumerates top k optimal trees that explain the input clusters.

## Requirements
- Python with `gurobipy` installed and a valid Gurobi license available in the environment (install via `pip install -r requirements.txt` once your Gurobi license is configured).

## Input Format
Tab-separated table with one mutation per row, all sample columns in the middle, and the cluster label in the final column:

```
mutID	sample0	sample1	clusterID
M1	0.40	0.30	clusterA
M2	0.18	0.52	clusterB
M3	0.11	0.44	clusterA
M4	0.10	0.20	clusterC
```

Mutation IDs must be unique. Cluster names can be any string. Clusters with `<= --min-mutations-per-cluster` mutations are dropped before solving; you can also exclude high-variance clusters via `--filter-unbiased-var-ge`.

## Example Run
```bash
python citup2.py \
  --frequenciesFile data/clusters.tsv \
  --pathOutputFilePrefix results/run1/cn_fit \
  --objective cluster \
```
Outputs will be written with the given prefix (e.g., `results/run1/cn_fit.tree_0.txt`).

## Description of Arguments/Parameters
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

## Simulations
You can generate synthetic inputs and tree visualizations with `simualtons/simulations.py`, which emits `mutations.tsv`, `clusters.tsv`, tree files, and a PNG in the chosen output directory. Example:
```bash
python simualtons/simulations.py --tree-size 6 --num-samples 8 --seed 3 --outdir simulations/out1 --run-name demo
```
Simulations used in the paper are located in:
`/nfshomes/hiseric/citup_recomb/CITUP2/simualtons/paper`
Use `--help` for options to add noise, simulate reads, or change tree size/coverage.
