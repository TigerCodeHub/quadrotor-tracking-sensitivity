"""Shared robust-regression helpers for the re-analysis (audit items #9/#10/#11/#12).

Pure numpy (runs on the py312/.venv analysis envs; no statsmodels needed). All fits are
OLS on config-mean rows; these helpers add what the original scripts lacked:

  * HC3 heteroskedasticity-consistent standard errors (not classical homoskedastic).
  * clustered pairs-bootstrap SEs (cluster = base config / airframe) for dependence.
  * importance by STANDARDIZED coefficient and by INTERQUARTILE effect, not raw |b|
    (raw coefficients have incompatible units/ranges -- audit #9).
  * drop-one delta-R^2 (R2_full - R2_without) and the conventional partial R^2
    ((R2_full - R2_reduced)/(1 - R2_reduced)) per factor.
  * explicit missing-row handling: callers drop non-finite predictor rows (audit #12),
    these helpers assume finite inputs.

Design note: at n~1e4-1e5 even tiny effects are "significant"; report the effect across a
factor's realistic range (IQR / full range) with a CI, not just p (audit #11).
"""
import numpy as np


def _hat_diag(X, XtX_inv):
    """Leverage h_ii = x_i' (X'X)^-1 x_i, vectorised (no n x n matrix)."""
    return np.einsum("ij,jk,ik->i", X, XtX_inv, X)


def ols_hc3(X, y):
    """OLS with HC3 robust covariance.

    Returns dict: beta, se (HC3), t, p (normal approx), r2, n, resid, XtX_inv.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    h = np.clip(_hat_diag(X, XtX_inv), 0.0, 1.0 - 1e-9)
    u = resid / (1.0 - h)                      # HC3 scaling
    meat = (X * (u ** 2)[:, None]).T @ X
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    from math import erf, sqrt
    p = np.array([2.0 * (1.0 - 0.5 * (1.0 + erf(abs(ti) / sqrt(2)))) if np.isfinite(ti)
                  else np.nan for ti in t])
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else float("nan")
    return {"beta": beta, "se": se, "t": t, "p": p, "r2": r2, "n": n,
            "resid": resid, "XtX_inv": XtX_inv}


def cluster_bootstrap_se(X, y, clusters, n_boot=1000, seed=12345):
    """Pairs cluster-bootstrap SE: resample whole clusters with replacement.

    clusters: 1-D array of cluster labels (e.g. base-config id or airframe). Returns
    (se_boot, beta_hat) over the coefficient vector.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    base = ols_hc3(X, y)["beta"]
    betas = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[c] for c in pick])
        Xb, yb = X[rows], y[rows]
        beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        betas[b] = beta
    return betas.std(axis=0, ddof=1), base


def importance(X, y, names, factor_cols):
    """Rank design factors by effect size, not raw |b| (audit #9).

    X includes an intercept at col 0; names are the column names (len == X.shape[1]);
    factor_cols lists the design-factor column names to rank. Returns a list of dicts
    sorted by |IQR effect| desc, each with standardized beta, IQR effect, drop-one delta-R^2 and partial R^2.

    NOTE: IQR is taken over the pooled predictor, so callers must pass predictors whose pooled
    spread equals the within-airframe spread the fixed-effect coefficient is identified from
    (e.g. log(1+payload_fraction), |com_y|/wheelbase), not raw log mass or CoM metres.
    """
    fit = ols_hc3(X, y)
    beta, r2_full = fit["beta"], fit["r2"]
    col = {nm: i for i, nm in enumerate(names)}
    ystd = y.std()
    out = []
    for f in factor_cols:
        i = col[f]
        xi = X[:, i]
        q1, q3 = np.percentile(xi, [25, 75])
        xstd = xi.std()
        # drop column i, refit, compare
        keep = [j for j in range(X.shape[1]) if j != i]
        r2_drop = ols_hc3(X[:, keep], y)["r2"]
        drop_one_dr2 = max(r2_full - r2_drop, 0.0)
        partial_r2 = drop_one_dr2 / (1.0 - r2_drop) if r2_drop < 1.0 else float("nan")
        out.append({
            "factor": f,
            "beta": float(beta[i]),
            "std_beta": float(beta[i] * xstd / ystd) if ystd > 0 else float("nan"),
            "iqr_effect": float(beta[i] * (q3 - q1)),   # log-units across IQR of x
            "drop_one_dr2": float(drop_one_dr2),
            "partial_r2": float(partial_r2),
            "t": float(fit["t"][i]),
            "p": float(fit["p"][i]),
        })
    out.sort(key=lambda d: -abs(d["iqr_effect"]))
    return out, r2_full


def finite_mask(*arrays):
    """Row mask where every array is finite (for dropping missing-predictor rows)."""
    m = np.ones(len(arrays[0]), bool)
    for a in arrays:
        m &= np.isfinite(np.asarray(a, float))
    return m


# ---------------------------------------------------------------------------------------
# Rank-aware HC3 fit + robust Wald test (audit pass 3 #7/#11). Added for the v3 mechanism
# scripts; does not change anything above.
# ---------------------------------------------------------------------------------------
def independent_columns(X, tol=1e-9):
    """Greedy Gram-Schmidt in column order: keep a column only if it is not (numerically) in
    the span of the columns already kept. Returns (kept_idx, dropped_idx). Earlier columns
    win, so put intercept/covariates first and the dummy block under test last."""
    X = np.asarray(X, float)
    kept, dropped = [], []
    Q = np.zeros((X.shape[0], 0))
    for j in range(X.shape[1]):
        v = X[:, j].copy()
        nv = np.linalg.norm(v)
        if nv == 0:
            dropped.append(j)
            continue
        r = v - Q @ (Q.T @ v) if Q.shape[1] else v
        r = r - Q @ (Q.T @ r) if Q.shape[1] else r      # re-orthogonalise
        if np.linalg.norm(r) / nv > tol:
            Q = np.column_stack([Q, r / np.linalg.norm(r)])
            kept.append(j)
        else:
            dropped.append(j)
    return kept, dropped


def ols_hc3_rank(X, y, names=None, tol=1e-9):
    """Rank-aware HC3 OLS. Drops aliased columns (reported), fits the full-rank remainder.

    Returns dict(beta, se, t, p, cov, r2, n, names (kept), aliased (dropped names),
    rank_full, cond (of the kept, column-scaled X), k_full).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    names = list(names) if names is not None else ["x%d" % i for i in range(X.shape[1])]
    kept, dropped = independent_columns(X, tol)
    Xk = X[:, kept]
    n, k = Xk.shape
    XtX_inv = np.linalg.inv(Xk.T @ Xk)
    beta = XtX_inv @ (Xk.T @ y)
    resid = y - Xk @ beta
    h = np.clip(_hat_diag(Xk, XtX_inv), 0.0, 1.0 - 1e-9)
    u = resid / (1.0 - h)
    meat = (Xk * (u ** 2)[:, None]).T @ Xk
    cov = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    from math import erf, sqrt
    p = np.array([2.0 * (1.0 - 0.5 * (1.0 + erf(abs(ti) / sqrt(2)))) if np.isfinite(ti)
                  else np.nan for ti in t])
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else float("nan")
    scale = np.linalg.norm(Xk, axis=0)
    sv = np.linalg.svd(Xk / scale, compute_uv=False)
    sv_full = np.linalg.svd(X / np.where(np.linalg.norm(X, axis=0) > 0,
                                         np.linalg.norm(X, axis=0), 1.0), compute_uv=False)
    rank_full = int((sv_full > sv_full[0] * 1e-9 * max(X.shape)).sum())
    return {"beta": beta, "se": se, "t": t, "p": p, "cov": cov, "r2": r2, "n": n,
            "names": [names[i] for i in kept], "aliased": [names[i] for i in dropped],
            "rank_full": rank_full, "k_full": X.shape[1],
            "cond": float(sv[0] / sv[-1]), "resid": resid}


def wald_block(fit, block_names):
    """Robust (HC3) joint Wald test that all coefficients in block_names (those kept in the
    fit) are zero. Returns (chi2, df, p, F=chi2/df)."""
    idx = [fit["names"].index(nm) for nm in block_names if nm in fit["names"]]
    if not idx:
        return float("nan"), 0, float("nan"), float("nan")
    b = fit["beta"][idx]
    V = fit["cov"][np.ix_(idx, idx)]
    chi2 = float(b @ np.linalg.solve(V, b))
    df = len(idx)
    try:
        from scipy.stats import chi2 as _c
        p = float(_c.sf(chi2, df))
    except ImportError:
        p = float("nan")
    return chi2, df, p, chi2 / df
