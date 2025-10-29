import numpy as np
from scipy.stats import spearmanr, kendalltau

# Example feature importance results
orig_importance = {'A': 0.86, 'B': 0.72, 'C': 0.40, 'D': 0.12}
mst_importance  = {'A': 0.83, 'B': 0.70, 'C': 0.15, 'D': 0.25}

# Sort features by importance (descending)
orig_ranked = sorted(orig_importance, key=orig_importance.get, reverse=True)
mst_ranked  = sorted(mst_importance, key=mst_importance.get, reverse=True)

# Convert to rank vectors
feature_list = list(orig_importance.keys())
orig_ranks = [orig_ranked.index(f) for f in feature_list]
mst_ranks  = [mst_ranked.index(f) for f in feature_list]

# Compute Spearman and Kendall
spearman_corr, _ = spearmanr(orig_ranks, mst_ranks)
kendall_corr, _ = kendalltau(orig_ranks, mst_ranks)

print("Spearman correlation:", spearman_corr)
print("Kendall tau:", kendall_corr)
