# === Drop-in replacement ===
# Keeps the same API & return values as the author's code, with optimized graph creation options.
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from typing import Callable, Optional

# -------------------------
# 1) Correlation generator
# -------------------------
def corr_generator_mst(df: pd.DataFrame, method: Callable = pearsonr) -> pd.DataFrame:
    """
    Generate a correlation matrix of a dataframe using the specified method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        method (Callable, optional): pearsonr, kendalltau, spearmanr. Defaults to pearsonr.

    Returns:
        pd.DataFrame: Correlation matrix (symmetric, dtype float).
    """
    if method not in [pearsonr, kendalltau, spearmanr]:
        raise ValueError("method should be pearsonr, kendalltau, or spearmanr")
    cols = df.columns
    m = len(cols)
    corr_df = pd.DataFrame(np.eye(m), index=cols, columns=cols, dtype=float)

    # Compute upper triangle then reflect
    for i in range(m):
        x = df[cols[i]].to_numpy()
        for j in range(i + 1, m):
            y = df[cols[j]].to_numpy()
            r, _ = method(x, y)
            corr_df.iat[i, j] = r
            corr_df.iat[j, i] = r
    return corr_df


# --------------------------------
# 2) Similarity/distance generator
# --------------------------------
def matrix_generator_mst(df: pd.DataFrame, method: Callable = pearsonr) -> pd.DataFrame:
    """
    Generate a similarity/distance matrix for a dataframe using the specified method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        method (Callable, optional): pearsonr, kendalltau, spearmanr,
            mutual_info_score, mutual_info_regression, or a custom callable f(x,y)->float.

    Returns:
        pd.DataFrame: Matrix of pairwise measures (symmetric if the method is symmetric).
    """
    cols = df.columns
    m = len(cols)
    M = pd.DataFrame(np.zeros((m, m), dtype=float), index=cols, columns=cols)

    # Case 1: standard correlations
    if method in [pearsonr, kendalltau, spearmanr]:
        return corr_generator_mst(df, method)

    # Case 2: mutual information for categorical variables
    if method == mutual_info_score:
        # heuristic: MI classifier best for low-cardinality columns
        if df.apply(lambda s: s.nunique(dropna=False)).max() > 10:
            raise ValueError("mutual_info_score is best for categorical data (≤10 unique values per column).")
        for i in range(m):
            xi = df[cols[i]]
            for j in range(i + 1, m):
                mj = method(xi, df[cols[j]])
                M.iat[i, j] = mj
                M.iat[j, i] = mj
        return M

    # Case 3: mutual information regression (asymmetric in general; we record both dirs)
    if method == mutual_info_regression:
        for i in range(m):
            xi = df[[cols[i]]]
            for j in range(m):
                if i == j:
                    continue
                yj = df[cols[j]]
                # returns array of shape (1,)
                val = method(xi, yj)
                M.iat[i, j] = float(val[0])
        return M

    # Case 4: custom callable f(x, y) -> float (assumed symmetric unless it’s KL, etc.)
    for i in range(m):
        xi = df[cols[i]].to_numpy()
        for j in range(i + 1, m):
            yj = df[cols[j]].to_numpy()
            val = method(xi, yj)
            M.iat[i, j] = float(val)
            # If the user supplies a non-symmetric method (e.g., KL), they should call twice externally.
            # Here we assume symmetry by default, mirroring to keep original API.
            M.iat[j, i] = float(val)
    return M


# ---------------------------------------------------
# 3) Minimal graph with optimized 'mst' & 'mst_knnX'
# ---------------------------------------------------
def create_minimal_edge_graph_mst(
    W: pd.DataFrame,
    version: str = 'v3',
    reverse: bool = True,
    verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a weight matrix to a minimal adjacency matrix that preserves connectivity.

    Args:
        W (pd.DataFrame): Weight matrix (any real values). We use abs(W) as edge strength, matching the paper code.
        version (str, optional):
            - 'v1': stop when all nodes have appeared at least once.
            - 'v2': continue until the graph is connected (then stop).
            - 'v3': add edges while graph is not connected with all nodes present (original behavior).
            - 'mst': maximum-spanning tree over |W| (fast, exactly M-1 edges, guaranteed connectivity if possible).
            - 'mst_knn1': MST + top-1 extra non-MST edge per node (if available).
            - 'mst_knn2': MST + top-2 extra non-MST edges per node (if available).
        reverse (bool, optional): Sort order for v1/v2/v3 edge sweep (True=descending). Ignored by 'mst*'.
        verbose (bool, optional): Print small diagnostics.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (adjacency_matrix, reduced_weight_matrix), both indexed/columned as W.
    """
    columns = list(W.columns)
    m = len(columns)
    absW = W.abs().copy()

    # Output frames
    adjacency_matrix = pd.DataFrame(0, index=columns, columns=columns, dtype=np.int8)
    reduced_df = pd.DataFrame(0.0, index=columns, columns=columns, dtype=np.float64)

    def _finalize_from_edges(edge_list):
        """Fill adjacency_matrix and reduced_df from iterable of (u,v,weight)."""
        for u, v, w in edge_list:
            adjacency_matrix.at[u, v] = 1
            adjacency_matrix.at[v, u] = 1
            reduced_df.at[u, v] = w
            reduced_df.at[v, u] = w

    # Fast path: MST family
    if version.startswith('mst'):
        # Build undirected graph with weights = abs(W)
        G = nx.Graph()
        for i, u in enumerate(columns):
            for j in range(i + 1, m):
                v = columns[j]
                w = float(absW.iat[i, j])
                if np.isfinite(w) and w > 0:
                    G.add_edge(u, v, weight=w)

        # Handle the case with isolated nodes or all-zero weights:
        if G.number_of_edges() == 0:
            if verbose:
                print("No positive edges; returning empty adjacency with zeros.")
            return adjacency_matrix, reduced_df

        # Compute maximum spanning tree (one per connected component), then merge
        mst_edges = []
        for comp in nx.connected_components(G):
            sub = G.subgraph(comp)
            T = nx.maximum_spanning_tree(sub, weight='weight')
            mst_edges.extend((u, v, float(d['weight'])) for u, v, d in T.edges(data=True))

        # Add MST edges to outputs
        _finalize_from_edges(mst_edges)

        # Optional k-NN augmentation on top of MST
        if version in ('mst_knn1', 'mst_knn2'):
            k_extra = 1 if version.endswith('knn1') else 2
            # Track existing undirected pairs
            existing = {tuple(sorted((u, v))) for u, v, _ in mst_edges}
            # For each node, add up to k strongest missing edges
            for u in columns:
                # Candidates not yet in MST
                cand = []
                ui = columns.index(u)
                for j, v in enumerate(columns):
                    if u == v:
                        continue
                    key = tuple(sorted((u, v)))
                    if key in existing:
                        continue
                    w = float(absW.iat[min(ui, j), max(ui, j)])
                    if np.isfinite(w) and w > 0:
                        cand.append((w, v))
                cand.sort(reverse=True, key=lambda t: t[0])
                added = 0
                for w, v in cand:
                    if added >= k_extra:
                        break
                    # Add edge
                    adjacency_matrix.at[u, v] = 1
                    adjacency_matrix.at[v, u] = 1
                    reduced_df.at[u, v] = w
                    reduced_df.at[v, u] = w
                    existing.add(tuple(sorted((u, v))))
                    added += 1

        if verbose:
            nnz = int(adjacency_matrix.values.sum() // 2)
            print(f"[{version}] edges selected: {nnz}")
        return adjacency_matrix, reduced_df

    # Original edge-scan behaviors (v1 / v2 / v3), preserved for compatibility
    edges = []
    for i in range(m):
        for j in range(i + 1, m):
            edges.append((columns[i], columns[j], float(absW.iat[i, j])))
    edges.sort(key=lambda x: x[2], reverse=reverse)

    connected_nodes = set()

    def is_graph_connected() -> bool:
        G = nx.Graph(adjacency_matrix)
        # nx.is_connected requires at least one node; when graph empty, treat as not connected
        return G.number_of_nodes() > 0 and nx.is_connected(G)

    for (u, v, w) in edges:
        add_edge = False

        if version == 'v1':
            if u not in connected_nodes or v not in connected_nodes:
                add_edge = True
                # If this edge brings in the final missing node, add and finish
                if len(connected_nodes.union({u, v})) == m:
                    adjacency_matrix.at[u, v] = 1
                    adjacency_matrix.at[v, u] = 1
                    reduced_df.at[u, v] = w
                    reduced_df.at[v, u] = w
                    if verbose:
                        print(f"v1 terminating at weight: {w}")
                    break

        elif version == 'v2':
            if u not in connected_nodes or v not in connected_nodes:
                add_edge = True
            elif len(connected_nodes) == m and not is_graph_connected():
                add_edge = True
            elif len(connected_nodes) == m and is_graph_connected():
                if verbose:
                    print(f"v2 terminating at weight: {w}")
                break

        elif version == 'v3':
            if not (len(connected_nodes) == m and is_graph_connected()):
                add_edge = True
            else:
                if verbose:
                    print(f"v3 terminating at weight: {w}")
                break

        if add_edge:
            adjacency_matrix.at[u, v] = 1
            adjacency_matrix.at[v, u] = 1
            reduced_df.at[u, v] = w
            reduced_df.at[v, u] = w
            connected_nodes.update([u, v])

    return adjacency_matrix, reduced_df
