# Example run

```bash
python citup2.py \
  --frequenciesFile example/input/example.tsv \
  --pathOutputFilePrefix example/output/example_out \
  --objective cluster \
```

Outputs will be written with the given prefix (e.g., `example/output/example_out.tree_0.txt`).


## Outputs exmaplanation
- `example_out.cluster_variance.tsv`: per-cluster counts and unbiased variance report after filtering.
- `example_out.gurobi.log`: solver log mirrored from Gurobi.
- `example_out.treeScores.tsv`: fit value and root degree for each enumerated tree; best-by-fit is marked.
- `example_out.tree_0.txt`: best tree structure, cluster/mutation assignments, node-cluster mappings, node/clade frequencies, and residuals for each solution.
