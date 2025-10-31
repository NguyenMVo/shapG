# benchmark_feature_importance.py
# ShapG+MST, CIS, SamplingSHAP, and LIME benchmarking
# - Feature-drop curves (like Fig. 8/9)
# - Weighted slope S (alpha=0.8)
# - Empty-feature baseline guard to avoid LightGBM crash

from __future__ import annotations
import os
import pickle
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone, is_classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

import lightgbm as lgb
import networkx as nx
from scipy.stats import spearmanr

from shapG.shapley import shapG, cis
import shapG.plot as shapGplot
from shapG.utils import *

###################################################################################
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


###################################################################################
# === Drop-in replacement ===
# Keeps the same API & return values as the author's code, with optimized graph creation options.
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from typing import Callable, Optional
from typing import Any, Tuple, Dict, List
# -------------------------
# 1) Correlation generator
# -------------------------

# ===== optional deps =====
try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

# ==== Your project modules (adjust paths if needed) ====
try:
    from cis import cis
except Exception:
    cis = None  # If CIS is not available, we will skip it safely.


# =============================================================================
# KPI HELPERS (classification accuracy / regression R^2), with empty-feature guard
# =============================================================================

def classification_kpi(X: pd.DataFrame, y: np.ndarray, S, *,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       model: lgb.LGBMClassifier | None = None) -> float:
    """
    Accuracy-based KPI for a subset of features S (classification).
    If S is empty, return majority-class accuracy (baseline).
    """
    cols = list(S)
    if len(cols) == 0:
        vals, counts = np.unique(y, return_counts=True)
        return float(np.max(counts) / len(y))

    if model is None:
        model = lgb.LGBMClassifier(learning_rate=0.05, verbosity=-1)

    x_train, x_test, y_train, y_test = train_test_split(
        X[cols], y, test_size=test_size, random_state=random_state, stratify=y
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return float(accuracy_score(y_test, y_pred))


def r2_kpi(X: pd.DataFrame, y: np.ndarray, S, *,
           test_size: float = 0.2,
           random_state: int = 42,
           model: lgb.LGBMRegressor | None = None) -> float:
    """
    R² KPI for a subset of features S (regression).
    If S is empty, return 0.0 (mean predictor baseline).
    """
    cols = list(S)
    if len(cols) == 0:
        return 0.0

    if model is None:
        model = lgb.LGBMRegressor(learning_rate=0.05, verbosity=-1)

    x_train, x_test, y_train, y_test = train_test_split(
        X[cols], y, test_size=test_size, random_state=random_state
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return float(r2_score(y_test, y_pred))


def _baseline_score_from_train(y_train: np.ndarray, y_val: np.ndarray, task: str) -> float:
    """
    When there are ZERO features, don't fit a model.
    - Classification: majority-class accuracy on the validation fold.
    - Regression: R^2 baseline ≈ 0.0
    """
    if task == "cls":
        vals, counts = np.unique(y_train, return_counts=True)
        maj = vals[np.argmax(counts)]
        return float(np.mean(y_val == maj))
    else:
        return 0.0


# =============================================================================
# Utility: robustly convert ShapG/CIS output to {feature_name: float_value}
# =============================================================================

def _to_phi_dict(shap_out: Any, feature_names: List[str]) -> Dict[str, float]:
    feats = list(feature_names)
    if isinstance(shap_out, dict):
        keys = list(shap_out.keys())
        # keyed by names
        if all(k in feats for k in keys):
            return {k: float(shap_out[k]) for k in keys}
        # keyed by indices
        try:
            idx = [int(k) for k in keys]
            if all(0 <= i < len(feats) for i in idx):
                return {feats[int(k)]: float(shap_out[k]) for k in keys}
        except Exception:
            pass
    arr = np.asarray(shap_out).reshape(-1)
    assert len(arr) == len(feats), "Output length != #features"
    return {f: float(v) for f, v in zip(feats, arr)}


# =============================================================================
# Global ranking helpers: SamplingSHAP and LIME
# =============================================================================

def global_ranking_sampling_shap(X: pd.DataFrame,
                                 y: np.ndarray,
                                 model,
                                 is_cls: bool,
                                 n_bg: int = 200,
                                 nsamples: int = 2048,
                                 random_state: int = 42) -> List[str]:
    """
    Compute global feature ranking using SHAP SamplingExplainer.
    Returns list of features sorted most->least important (mean |shap|).
    """
    if shap is None:
        print("[warn] shap not installed; skipping SamplingSHAP.")
        return []

    rng = np.random.RandomState(random_state)

    # Fit the model once on all features
    mdl = clone(model)
    mdl.fit(X, y)

    # Background subset for SHAP (NumPy arrays are safer with Sampling/Kernel explainers)
    bg_idx = rng.choice(len(X), size=min(n_bg, len(X)), replace=False)
    X_bg_np = X.iloc[bg_idx].to_numpy()
    X_np = X.to_numpy()

    # ---- Wrap the predictor so SHAP can set attributes on it ----
    class _PredictorWrapper:
        def __init__(self, mdl, is_cls):
            self.mdl = mdl
            self.is_cls = is_cls
            # SHAP may try to set this; make it writable
            self.feature_names_in_ = None

        def __call__(self, A):
            # A is a numpy array
            if self.is_cls:
                # return probabilities for Kernel/Sampling explainers
                return self.mdl.predict_proba(A)
            else:
                return self.mdl.predict(A)

    fwrap = _PredictorWrapper(mdl, is_cls)

    # Build explainer
    explainer = shap.SamplingExplainer(fwrap, X_bg_np, seed=random_state)

    # Explain a manageable subset (use background size for symmetry)
    explain_idx = rng.choice(len(X_np), size=min(len(X_np), n_bg), replace=False)
    shap_exp = explainer(X_np[explain_idx], nsamples=nsamples)

    # ---- Aggregate to global feature importance (mean |shap| per feature) ----
    # Handle multiple possible shapes from different SHAP versions.
    # Target: vals shape = (n_features,)
    n_features = X.shape[1]
    arr = shap_exp.values

    if isinstance(arr, list):
        # Old classification API: list of arrays [ (n_samples, n_features) per class ]
        vals = np.mean([np.abs(a).mean(axis=0) for a in arr], axis=0)
    else:
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == n_features:
            # (n_samples, n_features)
            vals = np.mean(np.abs(arr), axis=0)
        elif arr.ndim == 3:
            # Could be (n_samples, n_features, n_outputs) or (n_samples, n_outputs, n_features)
            axes = list(range(3))
            # find which axis is features
            feat_axis = [ax for ax in axes if arr.shape[ax] == n_features]
            if not feat_axis:
                raise ValueError(f"Unexpected SHAP values shape {arr.shape}; cannot locate feature axis={n_features}")
            feat_axis = feat_axis[0]
            reduce_axes = tuple(ax for ax in axes if ax != feat_axis)
            vals = np.mean(np.abs(arr), axis=reduce_axes)
        else:
            raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")

    phi = {f: float(v) for f, v in zip(X.columns, vals)}
    return sorted(phi, key=phi.get, reverse=True)

def global_ranking_lime(X: pd.DataFrame,
                        y: np.ndarray,
                        model,
                        is_cls: bool,
                        n_rows_lime: int = 300,
                        random_state: int = 42) -> List[str]:
    """
    Compute global feature ranking using LIME (mean |weight| per feature across explained rows).
    Returns list of features sorted most->least important.
    """
    if LimeTabularExplainer is None:
        print("[warn] lime not installed; skipping LIME.")
        return []
    rng = np.random.RandomState(random_state)

    # Fit model on all features to get a stable predictor
    mdl = clone(model)
    mdl.fit(X, y)

    # Build explainer
    X_np = X.values
    feature_names = list(X.columns)
    explainer = LimeTabularExplainer(
        training_data=X_np,
        feature_names=feature_names,
        class_names=np.unique(y) if is_cls else None,
        mode='classification' if is_cls else 'regression',
        discretize_continuous=False, # keep numeric as-is for tabular medical data
        random_state=random_state
    )

    # sample rows to explain
    idx = rng.choice(len(X), size=min(n_rows_lime, len(X)), replace=False)

    # aggregate absolute weights
    agg = np.zeros(X.shape[1], dtype=float)
    for i in idx:
        x0 = X_np[i]
        if is_cls:
            # choose class = model's predicted class for that instance
            c = int(mdl.predict(X_np[[i]])[0])
            exp = explainer.explain_instance(
                x0,
                mdl.predict_proba,
                labels=[c],
                num_features=X.shape[1]
            )
            weights = dict(exp.as_list(label=c))
        else:
            exp = explainer.explain_instance(
                x0,
                mdl.predict,
                num_features=X.shape[1]
            )
            weights = dict(exp.as_list())

        # Map weights back to feature indices by name
        for j, name in enumerate(feature_names):
            if name in weights:
                agg[j] += abs(weights[name])

    # mean |weight| per feature
    agg /= max(1, len(idx))
    phi = {f: float(v) for f, v in zip(feature_names, agg)}
    return sorted(phi, key=phi.get, reverse=True)


# =============================================================================
# Feature-drop benchmarking (authors' style) with baseline guard
# =============================================================================

def slope_score_S(curve: List[Tuple[int, float]], alpha: float = 0.8) -> float:
    """
    Weighted slope score (higher = better).
    Distribute each step's drop equally across the removed features at that step.
    """
    drops = []
    for k in range(1, len(curve)):
        r_prev, s_prev = curve[k - 1]
        r_cur, s_cur = curve[k]
        step = max(1, r_cur - r_prev)
        delta = max(0.0, s_prev - s_cur)
        drops.extend([delta / step] * step)
    if not drops:
        return 0.0
    weights = np.array([alpha ** i for i in range(len(drops))], dtype=float)
    return float(np.sum(weights * np.array(drops)))


def plot_KPI_comparison_by_dict(reader: Callable[[], Tuple[pd.DataFrame, np.ndarray]],
                                feature_rankings: Dict[str, List[str]],
                                model,
                                filename: str | None = None,
                                limit: int | None = None,
                                test_size: float = 0.2,
                                random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Authors-style benchmarking plot with:
      - Accuracy for classifiers, R^2 for regressors (auto).
      - Baseline guard when 0 features remain (no fitting, no crash).
      - Metric-aware cache filename.

    Returns: dict method -> {"Curve": [(num_removed, score), ...], "Slope": float}
    """
    X, y = reader()
    all_cols = list(X.columns)
    is_cls = is_classifier(model)
    metric_name = "$R^2$" if not is_cls else "Accuracy"

    # cache per metric + limit
    model_name = type(model).__name__
    results_file = f"{model_name}_{'R2' if not is_cls else 'ACC'}_limit{limit}_dict_results.pkl"

    if os.path.exists(results_file):
        try:
            with open(results_file, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                print(f"[cache] Using cached results: {results_file}")
                _replot_curves(cached, metric_name, filename)
                return cached
        except Exception:
            pass  # ignore corrupt cache

    # One stratified/regular split for all methods (comparable curves)
    stratify = y if is_cls else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    results: Dict[str, Dict[str, Any]] = {}

    for name, ranking in feature_rankings.items():
        # ensure ranking only includes real columns, and keep order
        ranking = [f for f in ranking if f in all_cols]
        if limit is None:
            limit_iter = len(ranking)
        else:
            limit_iter = min(limit, len(ranking))

        # Start with ALL features kept
        kept = list(all_cols)
        curve: List[Tuple[int, float]] = []

        # First point: 0 removed
        X_sub_tr = X_train[kept]
        X_sub_va = X_val[kept]
        if X_sub_tr.shape[1] == 0:
            score0 = _baseline_score_from_train(y_train, y_val, "cls" if is_cls else "reg")
        else:
            m0 = clone(model)
            m0.fit(X_sub_tr, y_train)
            pred0 = m0.predict(X_sub_va)
            score0 = float(accuracy_score(y_val, pred0)) if is_cls else float(r2_score(y_val, pred0))
        curve.append((0, score0))

        # Drop features progressively
        for r in range(1, limit_iter + 1):
            to_drop = set(ranking[:r])
            kept = [c for c in all_cols if c not in to_drop]

            X_sub_tr = X_train[kept]
            X_sub_va = X_val[kept]

            if X_sub_tr.shape[1] == 0:
                score = _baseline_score_from_train(y_train, y_val, "cls" if is_cls else "reg")
            else:
                m = clone(model)
                m.fit(X_sub_tr, y_train)
                pred = m.predict(X_sub_va)
                score = float(accuracy_score(y_val, pred)) if is_cls else float(r2_score(y_val, pred))

            curve.append((r, score))

        # Ensure last point for "all removed" if limit covered everything
        if limit is None or limit_iter == len(ranking):
            if len(curve) == 0 or curve[-1][0] != len(all_cols):
                score_last = _baseline_score_from_train(y_train, y_val, "cls" if is_cls else "reg")
                curve.append((len(all_cols), score_last))

        S = slope_score_S(curve, alpha=0.8)
        results[name] = {"Curve": curve, "Slope": S}

    # Plot curves
    _replot_curves(results, metric_name, filename)

    # Save cache
    try:
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
        print(f"[cache] Saved results -> {results_file}")
    except Exception:
        pass

    return results


def _replot_curves(results: Dict[str, Dict[str, Any]],
                   metric_name: str,
                   filename: str | None):
    plt.figure(figsize=(8, 5))
    for name, obj in results.items():
        xs, ys = zip(*obj["Curve"])
        plt.plot(xs, ys, marker='o', linewidth=2, label=f"{name} (S={obj['Slope']:.3f})")
    plt.xlabel("# features removed")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs #Features Removed")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"[plot] Saved: {filename}")
    plt.show()


# =============================================================================
# MAIN BENCHMARK: build MST graph, run ShapG, CIS, SamplingSHAP, LIME, plot
# =============================================================================

def benchmark_feature_importance(reader: Callable[[], Tuple[pd.DataFrame, np.ndarray]],
                                 model,
                                 filename: str | None = None,
                                 limit: int | None = 10,
                                 *,
                                 # explainer budgets (tune for your machine)
                                 sampling_shap_bg: int = 200,
                                 sampling_shap_nsamples: int = 2048,
                                 lime_rows: int = 300,
                                 random_state: int = 42):
    """
    Benchmark feature importance using ShapG+MST, CIS, SamplingSHAP, and LIME.

    - If `model` is a classifier, curves/metrics are in Accuracy (like Section 5).
    - If `model` is a regressor, curves/metrics are in R².
    """
    # ---- Load data ----
    X, y = reader()
    is_cls = isinstance(model, lgb.LGBMClassifier)

    # ---- Build correlation matrix & MST graph (Spearman) ----
    W = matrix_generator(X, method=spearmanr)
    A, _ = create_minimal_edge_graph(W, version='mst', verbose=True)
    G = nx.Graph(A)

    # ---- KPI selector to be consistent with the benchmark metric ----
    def f_only_S(S):
        return classification_kpi(X, y, S) if is_cls else r2_kpi(X, y, S)

    def f_with_G(Gin, S):
        return f_only_S(S)

    # ---- ShapG on MST (use same KPI as benchmark) ----
    try:
        shapg_vals = shapG(G, m=13, f=f_only_S, approximate_by_ratio=True)
    except TypeError:
        # Some builds expect f(G,S)
        shapg_vals = shapG(G, m=13, f=f_with_G, approximate_by_ratio=True)

    phi_mst = _to_phi_dict(shapg_vals, X.columns)
    rank_mst = sorted(phi_mst, key=lambda f: abs(phi_mst[f]), reverse=True)

    # ---- CIS baseline (if available) ----
    rank_cis = None
    if cis is not None:
        try:
            cis_vals = cis(G, f=f_with_G)  # many CIS impls expect f(G,S)
        except TypeError:
            cis_vals = cis(G, f=f_only_S)
        phi_cis = _to_phi_dict(cis_vals, X.columns)
        rank_cis = sorted(phi_cis, key=phi_cis.get, reverse=True)
    else:
        cis_vals = None
        print("[warn] CIS module not found; skipping CIS curve.")

    # ---- SamplingSHAP global ranking ----
    rank_sampling = global_ranking_sampling_shap(
        X, y, model, is_cls,
        n_bg=sampling_shap_bg,
        nsamples=sampling_shap_nsamples,
        random_state=random_state
    )

    # ---- LIME global ranking ----
    rank_lime = global_ranking_lime(
        X, y, model, is_cls,
        n_rows_lime=lime_rows,
        random_state=random_state
    )

    # ---- Assemble rankings dict for plotting ----
    feature_rankings = {'ShapG+MST': rank_mst}
    if rank_cis is not None:
        feature_rankings['CIS'] = rank_cis
    if len(rank_sampling) > 0:
        feature_rankings['SamplingSHAP'] = rank_sampling
    if len(rank_lime) > 0:
        feature_rankings['LIME'] = rank_lime

    # ---- Plot/compute curves (Accuracy or R² auto by model type) ----
    results = plot_KPI_comparison_by_dict(
        reader, feature_rankings, model, filename, limit
    )
    print("Top-10 ShapG+MST to be dropped first:", rank_mst[:10])
    print("Top-10 LIME to be dropped first:",      rank_lime[:10])
    print("Top-10 SamplingSHAP to be dropped first:", rank_sampling[:10])

    return shapg_vals, cis_vals, results


# =============================================================================
# OPTIONAL: dataset reader stub (keep yours if already defined elsewhere)
# =============================================================================
def h1n1_data_reader(filename: str = 'examples/data/process_data.csv',
                     target_col: str = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load dataset. If target_col is None, assume the last column is the label.
    Keep your own reader if you already have it in this file/project.
    """
    data = pd.read_csv(filename)
    X = data.drop(['h1n1_vaccine','respondent_id','seasonal_vaccine'],axis = 1)
    y =  data['h1n1_vaccine']
    return X, y

def diabetes_data_reader(filename: str = 'examples/data/diabetes_binary_health_indicators_BRFSS2015.csv',
                     target_col: str = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load dataset. If target_col is None, assume the last column is the label.
    Keep your own reader if you already have it in this file/project.
    """
    df = pd.read_csv(filename)
    if target_col is None:
        target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return X, y

def heart_disease_data_reader(filename='xamples/data/Heart_disease_cleveland_new.csv'):
    data = pd.read_csv(filename)
    X = data.drop(['target'],axis = 1)
    y =  data['target']
    return X, y


# =============================================================================
# Run directly: classification example (Accuracy curves like Section 5)
# =============================================================================
if __name__ == "__main__":
    # Use your real reader if present; this stub uses last column as label.
    reader = lambda: h1n1_data_reader('examples/data/process_data.csv')

    # Classification benchmark (Accuracy)
    model = lgb.LGBMClassifier(learning_rate=0.05, verbosity=-1)
    shapg_vals, cis_vals, results = benchmark_feature_importance(
        reader,
        model,
        filename="accuracy_benchmark_shapg_mst_vs_cis_sampling_lime.png",
        limit=None,                 # set an int (e.g., 15) to drop only first K features for speed
        sampling_shap_bg=200,       # tune for compute budget
        sampling_shap_nsamples=2048,
        lime_rows=300,
        random_state=42
    )
    print("Weighted slope S (alpha=0.8):",
          {k: v["Slope"] for k, v in results.items()})
    
