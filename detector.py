import numpy as np
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


def _sanitise(arr):
    """Replace NaN/Inf with finite fallbacks."""
    arr = np.asarray(arr, dtype=float)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr)
    fmax = arr[finite_mask].max()
    fmin = arr[finite_mask].min()
    arr = np.where(np.isnan(arr),    0.0,  arr)
    arr = np.where(np.isposinf(arr), fmax, arr)
    arr = np.where(np.isneginf(arr), fmin, arr)
    return arr


def _minmax(a):
    a = _sanitise(a)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def detect(X, z_threshold=3.0, eps=0.8, min_samples=5, contamination=0.05,
           X_benign=None):
    """
    X         — full feature matrix (all rows, benign + attack)
    X_benign  — benign-only feature matrix used to FIT the detectors.
                If None, falls back to fitting on all of X (original
                behaviour — weaker but still works).

    Fitting on benign-only data is the correct anomaly detection approach:
    the detectors learn what 'normal' looks like from clean data, then
    score everything including attacks against that baseline. When detectors
    fit on mixed data (attacks + benign), attacks define part of 'normal'
    and the scoring collapses.
    """
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    # Use benign-only subset for fitting if provided
    X_fit = np.asarray(X_benign, dtype=float) if X_benign is not None else X
    n_fit = X_fit.shape[0]

    # ── Z-Score ────────────────────────────────────────────────────────
    # Compute z-scores relative to the BENIGN distribution (mean/std from
    # X_fit), then score ALL rows against that benign baseline.
    benign_mean = X_fit.mean(axis=0)
    benign_std  = X_fit.std(axis=0)
    benign_std  = np.where(benign_std < 1e-10, 1.0, benign_std)  # avoid /0

    z_abs = np.abs((X - benign_mean) / benign_std)
    z_abs = np.nan_to_num(z_abs, nan=0.0, posinf=0.0, neginf=0.0)

    z_score_continuous = _sanitise(z_abs.max(axis=1))
    z_result = (z_abs > z_threshold).any(axis=1).astype(int)

    # ── DBSCAN ────────────────────────────────────────────────────────
    # Fit DBSCAN on benign data to establish cluster structure, then
    # predict on all data — points outside benign clusters = anomalies.
    db = DBSCAN(eps=min(eps, 0.5), min_samples=max(2, min_samples // 2))
    db.fit(X_fit)

    # Assign each point in X to its nearest cluster core point.
    # Points with no core point within eps are labelled -1 (anomaly).
    if len(db.core_sample_indices_) > 0:
        from sklearn.neighbors import NearestNeighbors as _NN
        core_pts = X_fit[db.core_sample_indices_]
        nn = _NN(n_neighbors=1).fit(core_pts)
        dists, _ = nn.kneighbors(X)
        db_result = (dists[:, 0] > eps).astype(int)
        db_score_continuous = _sanitise(dists[:, 0])
    else:
        # No clusters found — flag everything
        db_result = np.ones(n_samples, dtype=int)
        db_score_continuous = np.ones(n_samples)

    # ── Isolation Forest ──────────────────────────────────────────────
    # Fit on benign data only — contamination is 0 conceptually since
    # X_fit contains only benign rows, but IsolationForest requires a
    # small positive value. Use a data-driven estimate from z-scores.
    auto_contam = float(np.clip(
        (z_abs > z_threshold).any(axis=1).mean(),
        0.01, 0.499
    ))

    iso = IsolationForest(
        n_estimators=200,
        contamination=auto_contam,
        max_samples=min(256, n_fit),
        random_state=42,
    )
    iso.fit(X_fit)
    iso_result = np.where(iso.predict(X) == -1, 1, 0)
    iso_score_continuous = _sanitise(-iso.decision_function(X))

    # Auto-flip: if iso or dbscan scores are anti-correlated with the
    # z-score signal, flip them — ensures all scores point in the same
    # direction (higher = more anomalous) for a valid ensemble average.
    if np.corrcoef(iso_score_continuous, z_score_continuous)[0, 1] < 0:
        iso_score_continuous = _sanitise(iso_score_continuous.max() - iso_score_continuous)
    if np.corrcoef(db_score_continuous, z_score_continuous)[0, 1] < 0:
        db_score_continuous  = _sanitise(db_score_continuous.max() - db_score_continuous)

    # ── Ensemble ──────────────────────────────────────────────────────
    votes    = z_result + db_result + iso_result
    ensemble = np.where(votes >= 2, 1, 0)

    ensemble_score_continuous = _sanitise(
        (_minmax(z_score_continuous)
         + _minmax(db_score_continuous)
         + _minmax(iso_score_continuous)) / 3.0
    )

    continuous_scores = {
        "Z-Score":          z_score_continuous,
        "DBSCAN":           db_score_continuous,
        "Isolation Forest": iso_score_continuous,
        "Ensemble":         ensemble_score_continuous,
    }

    return z_result, db_result, iso_result, votes, ensemble, continuous_scores
