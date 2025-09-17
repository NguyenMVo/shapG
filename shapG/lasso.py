import numpy as np
import pandas as pd
import networkx as nx
from typing import Callable, Optional, Union
from scipy.stats import pearsonr, kendalltau, spearmanr, rankdata, norm
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.covariance import GraphicalLasso, graphical_lasso

# ===================== helpers (suffixed) =====================

def _gaussian_rank_transform_col_lasso(x: np.ndarray) -> np.ndarray:
    """Ranks -> uniform -> normal scores (van der Waerden)."""
    r = rankdata(x, method="average")
    u = (r - 0.5) / len(x)
    return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))

def _nonparanormal_transform_lasso(X: np.ndarray) -> np.ndarray:
    return np.column_stack([_gaussian_rank_transform_col_lasso(X[:, j]) for j in range(X.shape[1])])

def _standardize_lasso(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, ddof=1, keepdims=True)
    return (X - mu) / (sd + 1e-12)

def _adjacency_from_precision_lasso(Theta: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    A = (np.abs(Theta) > tol).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return A

def _partial_corr_from_precision_lasso(Theta: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(Theta), 1e-12, None))
    P = -Theta / np.outer(d, d)
    np.fill_diagonal(P, 1.0)
    return P

# ===================== StARS + GLasso (suffixed) =====================

def stars_glasso_corr_generator_lasso(
    df: pd.DataFrame,
    *,
    B: int = 50,
    subsample_frac: float = 0.8,
    beta: float = 0.05,
    alphas: Optional[np.ndarray] = None,
    nonparanormal: bool = True,
    impute: str = "median",
    random_state: Optional[int] = 0,
    max_iter: int = 100,
    return_probs: bool = False
) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    rng = np.random.default_rng(random_state)
    X = df.to_numpy(dtype=float)

    # Handle missing
    if np.isnan(X).any():
        if impute == "median":
            med = np.nanmedian(X, axis=0, keepdims=True)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(med, inds[1])
        elif impute == "drop":
            X = X[~np.any(np.isnan(X), axis=1)]
        else:
            raise ValueError("impute must be 'median' or 'drop'")

    # Rank->normal transform then standardize
    if nonparanormal:
        X = _nonparanormal_transform_lasso(X)
    X = _standardize_lasso(X)

    n, p = X.shape
    if p < 2:
        out = pd.DataFrame(np.ones((p, p)), index=df.columns, columns=df.columns)
        return (out, out.copy()) if return_probs else out

    # Penalty grid
    S_full = np.cov(X, rowvar=False)
    # robust lam_max from strictly off-diagonal entries
    off = S_full.copy()
    np.fill_diagonal(off, 0.0)
    lam_max = float(np.max(np.abs(off))) or 1.0
    if alphas is None:
        # large -> small (geomspace ensures descending if start > end)
        alphas = np.geomspace(lam_max, lam_max / 100.0, 15)

    edge_counts = np.zeros((len(alphas), p, p), dtype=np.float64)

    # -------- StARS subsampling --------
    for _ in range(B):
        idx = rng.choice(n, size=int(np.floor(subsample_frac * n)), replace=False)
        Xb = X[idx, :]
        S_b = np.cov(Xb, rowvar=False)

        cov_init = None
        prec_init = None
        for ai, alpha in enumerate(alphas):
            # graphical_lasso returns (covariance, precision)
            try:
                cov_est, prec_est = graphical_lasso(
                    emp_cov=S_b, alpha=alpha,
                    cov_init=cov_init, precision_init=prec_init,
                    max_iter=max_iter, mode="cd", tol=1e-4, enet_tol=1e-4
                )
            except TypeError:
                # Your sklearn doesn't support cov_init/precision_init → cold-start
                cov_est, prec_est = graphical_lasso(
                    S_b, alpha, mode="cd", tol=1e-4, enet_tol=1e-4, max_iter=max_iter
                )

            A = _adjacency_from_precision_lasso(prec_est)
            edge_counts[ai] += A

            # Warm-start for next alpha if your version supports it in the try branch
            cov_init, prec_init = cov_est, prec_est

    # Instability path
    pis = edge_counts / B
    iu, ju = np.triu_indices(p, k=1)
    V_path = np.array([
        (2.0 / (p * (p - 1))) * np.sum(pis[ai][iu, ju] * (1 - pis[ai][iu, ju]))
        for ai in range(len(alphas))
    ])
    feasible = np.where(V_path <= beta)[0]
    a_star_idx = feasible[0] if feasible.size else int(np.argmin(V_path))
    alpha_star = float(alphas[a_star_idx])

    # -------- Final refit on all data --------
    try:
        cov_F, prec_F = graphical_lasso(
            emp_cov=S_full, alpha=alpha_star,
            max_iter=max_iter, mode="cd", tol=1e-4, enet_tol=1e-4
        )
    except TypeError:
        cov_F, prec_F = graphical_lasso(
            S_full, alpha_star, mode="cd", tol=1e-4, enet_tol=1e-4, max_iter=max_iter
        )

    P = _partial_corr_from_precision_lasso(prec_F)
    P_df = pd.DataFrame(P, index=df.columns, columns=df.columns)

    if not return_probs:
        return P_df

    pi_star = pis[a_star_idx]
    np.fill_diagonal(pi_star, 1.0)
    Pi_df = pd.DataFrame(pi_star, index=df.columns, columns=df.columns)
    return P_df, Pi_df

# ===================== optimized correlation (suffixed) =====================

def corr_generator_lasso(df: pd.DataFrame, method: Callable = pearsonr) -> pd.DataFrame:
    """Fast Pearson/Spearman correlation (exact), Kendall via pairwise loop.

    - Pearson: z-score once → GEMM
    - Spearman: rank each column once → z-score ranks → GEMM
    - Kendall: pairwise kendalltau
    """
    if method not in [pearsonr, kendalltau, spearmanr]:
        raise ValueError("method should be pearsonr, kendalltau, or spearmanr")

    X = df.to_numpy(dtype=float)
    n, p = X.shape

    # Fast Pearson
    if method is pearsonr:
        Z = (X - X.mean(0, keepdims=True)) / (X.std(0, ddof=1, keepdims=True) + 1e-12)
        W = (Z.T @ Z) / (n - 1)
        np.fill_diagonal(W, 1.0)
        return pd.DataFrame(W, index=df.columns, columns=df.columns)

    # Fast Spearman
    if method is spearmanr:
        R = np.column_stack([rankdata(X[:, j], method="average") for j in range(p)]).astype(float)
        Z = (R - R.mean(0, keepdims=True)) / (R.std(0, ddof=1, keepdims=True) + 1e-12)
        W = (Z.T @ Z) / (n - 1)
        np.fill_diagonal(W, 1.0)
        return pd.DataFrame(W, index=df.columns, columns=df.columns)

    # Kendall: pairwise
    corr_df = pd.DataFrame(np.eye(p), columns=df.columns, index=df.columns, dtype=float)
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i+1:]:
            corr, _ = kendalltau(df[col1], df[col2])
            corr_df.loc[col1, col2] = corr
            corr_df.loc[col2, col1] = corr
    return corr_df

# ===================== matrix router (suffixed) =====================

def matrix_generator_lasso(
    df: pd.DataFrame,
    method: Union[Callable, str] = pearsonr,
    **kwargs
) -> pd.DataFrame:
    """Router to build similarity matrices.

    method options:
      - pearsonr, kendalltau, spearmanr  -> correlation matrices (fast paths for Pearson/Spearman)
      - 'stars_glasso'                   -> rank-based partial correlation via StARS + Graphical Lasso
      - mutual_info_score                -> MI for categorical columns (≤10 unique values)
      - mutual_info_regression           -> MI regression (asymmetric; beware interpretability)
      - any other callable 'method(x, y)' -> generic pairwise (assumed symmetric)
    Extra kwargs are forwarded to the underlying method.
    """
    if isinstance(method, str) and method.lower() == "stars_glasso":
        return stars_glasso_corr_generator_lasso(df, **kwargs)  # DataFrame

    if method in [pearsonr, kendalltau, spearmanr]:
        return corr_generator_lasso(df, method)

    if method == mutual_info_score:
        # Categorical MI
        if df.apply(lambda x: len(pd.unique(x.dropna()))).max() > 10:
            raise ValueError("mutual_info_score is suited for categorical data (≤10 unique values per column)")
        matrix_df = pd.DataFrame(0.0, index=df.columns, columns=df.columns)
        for i, c1 in enumerate(df.columns):
            for c2 in df.columns[i+1:]:
                mi = mutual_info_score(df[c1], df[c2])
                matrix_df.loc[c1, c2] = matrix_df.loc[c2, c1] = mi
        return matrix_df

    if method == mutual_info_regression:
        # Asymmetric MI regression (fill full matrix for convenience)
        matrix_df = pd.DataFrame(0.0, index=df.columns, columns=df.columns)
        for c1 in df.columns:
            for c2 in df.columns:
                if c1 != c2:
                    measures = mutual_info_regression(df[[c1]], df[c2])
                    matrix_df.loc[c1, c2] = measures[0]
        return matrix_df

    # Generic callable fallback
    matrix_df = pd.DataFrame(0.0, index=df.columns, columns=df.columns)
    for i, c1 in enumerate(df.columns):
        for c2 in df.columns[i+1:]:
            val = method(df[c1], df[c2], **kwargs) if callable(method) else np.nan
            matrix_df.loc[c1, c2] = val
            matrix_df.loc[c2, c1] = val
    return matrix_df

# ===================== graph reducer (suffixed) =====================

def create_minimal_edge_graph_lasso(
    W: pd.DataFrame,
    version: str = 'v3',
    reverse: bool = True,
    verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a weight matrix to a minimal adjacency matrix that preserves connectivity."""
    columns = W.columns.tolist()

    # Build sorted edge list
    edges = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            edges.append((columns[i], columns[j], abs(W.iloc[i, j])))
    edges.sort(key=lambda x: x[2], reverse=reverse)

    connected_nodes = set()
    adjacency_matrix = pd.DataFrame(0, index=columns, columns=columns, dtype=np.int8)
    reduced_df = pd.DataFrame(0.0, index=columns, columns=columns, dtype=np.float64)

    def is_graph_connected_lasso():
        G = nx.Graph(adjacency_matrix)
        return nx.is_connected(G)

    for node1, node2, weight in edges:
        add_edge = False

        if version == 'v1':
            if node1 not in connected_nodes or node2 not in connected_nodes:
                add_edge = True
                if len(connected_nodes.union({node1, node2})) == len(columns):
                    if verbose:
                        print(f"v1 terminating at weight: {weight}")
                    adjacency_matrix.loc[node1, node2] = adjacency_matrix.loc[node2, node1] = 1
                    reduced_df.loc[node1, node2] = reduced_df.loc[node2, node1] = weight
                    break

        elif version == 'v2':
            if node1 not in connected_nodes or node2 not in connected_nodes:
                add_edge = True
            elif len(connected_nodes) == len(columns) and not is_graph_connected_lasso():
                add_edge = True
            elif len(connected_nodes) == len(columns) and is_graph_connected_lasso():
                if verbose:
                    print(f"v2 terminating at weight: {weight}")
                break

        elif version == 'v3':
            if not (len(connected_nodes) == len(columns) and is_graph_connected_lasso()):
                add_edge = True
            else:
                if verbose:
                    print(f"v3 terminating at weight: {weight}")
                break

        if add_edge:
            adjacency_matrix.loc[node1, node2] = adjacency_matrix.loc[node2, node1] = 1
            reduced_df.loc[node1, node2] = reduced_df.loc[node2, node1] = weight
            connected_nodes.update([node1, node2])

    return adjacency_matrix, reduced_df
