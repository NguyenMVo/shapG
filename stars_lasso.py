import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
from scipy.stats import rankdata, norm

# ---------- helpers ----------
def _gaussian_rank_transform_col(x: np.ndarray) -> np.ndarray:
    # midranks -> (0,1) -> normal scores (nonparanormal)
    r = rankdata(x, method="average")
    u = (r - 0.5) / len(x)
    return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))

def _nonparanormal_transform(X: np.ndarray) -> np.ndarray:
    return np.column_stack([_gaussian_rank_transform_col(X[:, j]) for j in range(X.shape[1])])

def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, ddof=1, keepdims=True)
    return (X - mu) / (sd + 1e-12)

def _adjacency_from_precision(Theta: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    A = (np.abs(Theta) > tol).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return A

# ---------- main: StARS + Graphical Lasso ----------
def stars_glasso_corr_generator(
    df: pd.DataFrame,
    *,
    B: int = 50,                 # number of subsamples
    subsample_frac: float = 0.8, # τ
    beta: float = 0.05,          # instability target
    alphas: np.ndarray | None = None,
    nonparanormal: bool = True,  # rank->normal transform before GLasso
    impute: str = "median",      # "median" | "drop"
    random_state: int | None = 0,
    max_iter: int = 100,
    return_probs: bool = False   # also return selection-prob matrix as DataFrame
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a partial-correlation matrix via StARS + Graphical Lasso.
    Returns a DataFrame with the same index/columns as df ([-1,1], diag=1).
    If return_probs=True, also returns a DataFrame of edge selection probabilities in [0,1].
    """
    rng = np.random.default_rng(random_state)
    X = df.to_numpy(dtype=float)

    # Handle missing values
    if np.isnan(X).any():
        if impute == "median":
            med = np.nanmedian(X, axis=0, keepdims=True)
            # simple median impute
            inds = np.where(np.isnan(X))
            X[inds] = np.take(med, inds[1])
        elif impute == "drop":
            mask = ~np.any(np.isnan(X), axis=1)
            X = X[mask]
        else:
            raise ValueError("impute must be 'median' or 'drop'")

    # Optional nonparanormal transform, then standardize
    if nonparanormal:
        X = _nonparanormal_transform(X)
    X = _standardize(X)

    n, p = X.shape
    if p < 2:
        out = pd.DataFrame(np.ones((p, p)), index=df.columns, columns=df.columns)
        return (out, out.copy()) if return_probs else out

    # Penalty grid
    S = np.cov(X, rowvar=False)  # (n-1) normalization
    lam_max = float(np.max(np.abs(S - np.diag(np.diag(S))))) or 1.0
    if alphas is None:
        alphas = np.geomspace(lam_max, lam_max / 100.0, 15)

    edge_counts = np.zeros((len(alphas), p, p), dtype=np.float64)

    # Subsampling loop
    for _ in range(B):
        idx = rng.choice(n, size=int(np.floor(subsample_frac * n)), replace=False)
        Xb = X[idx, :]

        # Iterate over alpha grid (avoid warm_start for sklearn compatibility)
        for a_idx, alpha in enumerate(alphas):
            model = GraphicalLasso(alpha=alpha, max_iter=max_iter)
            model.fit(Xb)

            A = _adjacency_from_precision(model.precision_)
            edge_counts[a_idx] += A

    # StARS instability path
    pis = edge_counts / B  # selection probabilities per alpha
    iu, ju = np.triu_indices(p, k=1)
    V_path = np.array([
        (2.0 / (p * (p - 1))) * np.sum(pis[a_idx][iu, ju] * (1 - pis[a_idx][iu, ju]))
        for a_idx in range(len(alphas))
    ])

    # choose smallest alpha with instability <= beta (or min V)
    feasible = np.where(V_path <= beta)[0]
    a_star_idx = feasible[0] if feasible.size else int(np.argmin(V_path))
    alpha_star = float(alphas[a_star_idx])

    # Final refit on all data
    final = GraphicalLasso(alpha=alpha_star, max_iter=max_iter).fit(X)
    Theta = final.precision_
    d = np.sqrt(np.clip(np.diag(Theta), 1e-12, None))
    P = -Theta / np.outer(d, d)
    np.fill_diagonal(P, 1.0)

    # Wrap up as DataFrames
    pcorr_df = pd.DataFrame(P, index=df.columns, columns=df.columns)
    if not return_probs:
        return pcorr_df

    pi_star = pis[a_star_idx]
    # Put 1 on diagonal to mirror "certainty" for self-edges (optional cosmetic)
    np.fill_diagonal(pi_star, 1.0)
    prob_df = pd.DataFrame(pi_star, index=df.columns, columns=df.columns)
    return pcorr_df, prob_df
