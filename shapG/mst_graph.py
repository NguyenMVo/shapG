# ============================================================
# Heavier MST + KNN + Degree Floors + Triangle Closing (drop-in)
# ============================================================
import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple

def create_minimal_edge_graph_mst_graph(
    W: pd.DataFrame,
    reverse: bool = True,                 # kept for API compatibility; not used
    version: str = 'mst_dense',           # new "heavy" default
    pct_tau: float = 0.92,                # global strength threshold (0.90–0.95 works well)
    knn_k: int = 8,                       # per-node extra edges (6–10 recommended)
    avg_degree_floor: float = 3.2,        # target average degree (3.0–3.6)
    min_degree_floor: int = 2,            # ensure no node is too isolated (2 or 3)
    triangle_closing: bool = True,        # add edges among neighbors to raise clustering
    triangle_budget_per_node: int = 3,    # limit triangle-closing edges per node
    max_global_edges: int = None,         # optional global cap; None = no cap
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a *dense* graph from a weight matrix W (heavier than MST/KNN),
    prioritizing strong edges and local synergy. Returns:
        adjacency_matrix (0/1 pd.DataFrame), reduced_weight_matrix (float pd.DataFrame)
    """

    # ---- setup
    cols = list(W.columns)
    m = len(cols)
    absW = W.abs().copy()

    A = pd.DataFrame(0, index=cols, columns=cols, dtype=np.int8)
    R = pd.DataFrame(0.0, index=cols, columns=cols, dtype=float)

    # ---- helper lambdas
    def _add_edge(u, v, w, existing):
        if u == v: return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing: return False
        A.at[u, v] = A.at[v, u] = 1
        R.at[u, v] = R.at[v, u] = w
        existing.add((a, b))
        return True

    def _avg_deg():
        return float(A.values.sum()) / m

    def _deg(u):
        return int(A.loc[u].sum())

    # ---- 1) MST backbone (maximum spanning forest over |W|)
    Gfull = nx.Graph()
    for i, ui in enumerate(cols):
        for j in range(i + 1, m):
            vj = cols[j]
            w = float(absW.iat[i, j])
            if np.isfinite(w) and w > 0:
                # deterministic tie-breaking with node names
                Gfull.add_edge(ui, vj, weight=w, tie=(min(ui, vj), max(ui, vj)))

    mst_edges = []
    for comp in nx.connected_components(Gfull):
        sub = Gfull.subgraph(comp)
        T = nx.maximum_spanning_tree(sub, weight='weight')
        mst_edges.extend((u, v, float(d['weight'])) for u, v, d in T.edges(data=True))

    existing = set()
    for u, v, w in mst_edges:
        _add_edge(u, v, w, existing)

    # Early exit: pure MST
    if version == 'mst':
        if verbose:
            print(f"[mst] edges={int(A.values.sum()//2)} avg_deg={_avg_deg():.2f}")
        return A, R

    # ---- collect all *non-MST* candidates (descending by weight; deterministic)
    cands = []
    for i, ui in enumerate(cols):
        for j in range(i + 1, m):
            vj = cols[j]
            key = (ui, vj) if ui < vj else (vj, ui)
            if key in existing:
                continue
            w = float(absW.iat[i, j])
            if np.isfinite(w) and w > 0:
                cands.append((w, ui, vj))
    if not cands:
        if verbose: print("[mst_dense] no non-MST candidates; returning MST.")
        return A, R

    # sort descending by w; tie-break lexicographically on (u,v) for determinism
    cands.sort(key=lambda t: (-t[0], min(t[1], t[2]), max(t[1], t[2])))
    weights = np.array([w for (w, _, _) in cands])
    tau = float(np.quantile(weights, pct_tau))

    # ---- 2) Global strength pass: keep all edges >= tau
    added_global = 0
    for (w, u, v) in cands:
        if w < tau: break
        if max_global_edges is not None and (A.values.sum()//2) >= max_global_edges:
            break
        added_global += _add_edge(u, v, w, existing)
    if verbose:
        print(f"[mst_dense] +global(≥τ={pct_tau:.2f})={added_global} "
              f"edges={int(A.values.sum()//2)} avg_deg={_avg_deg():.2f}")

    # ---- 3) Per-node kNN pass (top-K strongest missing edges per node)
    # Build a per-node candidate list (still sorted by strength)
    per_node = {u: [] for u in cols}
    for (w, u, v) in cands:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in existing: 
            continue
        per_node[u].append((w, v))
        per_node[v].append((w, u))

    added_knn = 0
    for u in cols:
        picks = 0
        for (w, v) in per_node[u]:
            if picks >= knn_k: 
                break
            if max_global_edges is not None and (A.values.sum()//2) >= max_global_edges:
                break
            if _add_edge(u, v, w, existing):
                picks += 1
                added_knn += 1
    if verbose:
        print(f"[mst_dense] +knn(k={knn_k})={added_knn} "
              f"edges={int(A.values.sum()//2)} avg_deg={_avg_deg():.2f}")

    # ---- 4) Degree floors
    # 4a) Min degree
    added_min_deg = 0
    if min_degree_floor is not None and min_degree_floor > 0:
        # For nodes below the floor, add their strongest missing edges
        for u in cols:
            if _deg(u) >= min_degree_floor:
                continue
            for (w, v) in per_node[u]:
                if _deg(u) >= min_degree_floor:
                    break
                if max_global_edges is not None and (A.values.sum()//2) >= max_global_edges:
                    break
                if _add_edge(u, v, w, existing):
                    added_min_deg += 1

    # 4b) Average degree (coarse loop until floor met or no progress)
    added_avg_deg = 0
    guard = 0
    while _avg_deg() < avg_degree_floor and guard < 5:
        progressed = False
        for (w, u, v) in cands:
            if max_global_edges is not None and (A.values.sum()//2) >= max_global_edges:
                break
            if _add_edge(u, v, w, existing):
                added_avg_deg += 1
                progressed = True
                if _avg_deg() >= avg_degree_floor:
                    break
        if not progressed:
            break
        guard += 1

    if verbose:
        print(f"[mst_dense] +min_deg={added_min_deg}, +avg_deg={added_avg_deg} "
              f"edges={int(A.values.sum()//2)} avg_deg={_avg_deg():.2f}")

    # ---- 5) Triangle closing (optional)
    if triangle_closing:
        added_tri = 0
        # For each node, try to connect its strongest unconnected neighbor pairs
        for u in cols:
            if triangle_budget_per_node <= 0:
                continue
            # neighbors of u
            Nu = [v for v in cols if A.at[u, v] == 1]
            if len(Nu) < 2:
                continue
            # consider pairs (a,b) in Nu not connected yet
            # score them by combined edge strength w(u,a)+w(u,b)+w(a,b)
            budget = triangle_budget_per_node
            pairs = []
            for i in range(len(Nu)):
                a = Nu[i]
                ai = cols.index(a)
                for j in range(i + 1, len(Nu)):
                    b = Nu[j]
                    if A.at[a, b] == 1:
                        continue
                    bi = cols.index(b)
                    wua = float(absW.iat[min(cols.index(u), ai), max(cols.index(u), ai)])
                    wub = float(absW.iat[min(cols.index(u), bi), max(cols.index(u), bi)])
                    wab = float(absW.iat[min(ai, bi), max(ai, bi)])
                    score = wua + wub + wab  # simple synergy proxy
                    pairs.append((score, wab, a, b))
            pairs.sort(reverse=True)  # highest synergy first
            for _, wab, a, b in pairs:
                if budget <= 0:
                    break
                if _add_edge(a, b, wab, existing):
                    added_tri += 1
                    budget -= 1
    else:
        added_tri = 0

    if verbose:
        print(f"[mst_dense] +triangles={added_tri} "
              f"FINAL edges={int(A.values.sum()//2)} avg_deg={_avg_deg():.2f}")

    return A, R
