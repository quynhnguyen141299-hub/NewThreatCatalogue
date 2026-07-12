import numpy as np
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


def _sanitise(arr):
    """Replace NaN/Inf with finite fallbacks so roc_curve never crashes."""
    arr = np.asarray(arr, dtype=float)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr)
    fmax = arr[finite_mask].max()
    fmin = arr[finite_mask].min()
    arr = np.where(np.isnan(arr),     0.0,  arr)
    arr = np.where(np.isposinf(arr),  fmax, arr)
    arr = np.where(np.isneginf(arr),  fmin, arr)
    return arr


def _minmax(a):
    a = _sanitise(a)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def detect(X, z_threshold=3.0, eps=0.8, min_samples=5, contamination=0.05):

    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    # ─────────────────────────────────────────────
    # Z-Score
    # ─────────────────────────────────────────────
    # Use absolute z-scores. Zero-variance columns produce NaN — replace
    # with 0 so they contribute nothing to the max rather than poisoning it.
    z_raw = zscore(X, axis=0)
    z_raw = np.nan_to_num(z_raw, nan=0.0, posinf=0.0, neginf=0.0)
    z_abs = np.abs(z_raw)

    # Continuous score: max |z| across all features per row.
    # Higher value = further from normal = more anomalous. ✓
    z_score_continuous = _sanitise(z_abs.max(axis=1))
    z_result = (z_abs > z_threshold).any(axis=1).astype(int)

    # ─────────────────────────────────────────────
    # DBSCAN
    # ─────────────────────────────────────────────
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    db_result = np.where(labels == -1, 1, 0)

    # Continuous score: mean distance to k nearest neighbours.
    # Points far from their neighbours are likely noise/anomalies.
    # Higher value = more isolated = more anomalous. ✓
    k = max(1, min(min_samples, n_samples - 1))
    if n_samples > 1:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        db_score_continuous = distances[:, 1:].mean(axis=1)
    else:
        db_score_continuous = np.zeros(n_samples)
    db_score_continuous = _sanitise(db_score_continuous)

    # ─────────────────────────────────────────────
    # Isolation Forest
    # ─────────────────────────────────────────────
    # Use a lower contamination to avoid over-flagging benign transactions,
    # and more estimators for a more stable decision boundary.
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
    )
    iso.fit(X)
    iso_result = np.where(iso.predict(X) == -1, 1, 0)

    # Continuous score: negate decision_function so higher = more anomalous.
    # decision_function returns LOWER values for anomalies (more negative),
    # so negating flips the direction: anomalies get HIGH positive scores. ✓
    iso_score_continuous = _sanitise(-iso.decision_function(X))

    # ─────────────────────────────────────────────
    # Ensemble vote
    # ─────────────────────────────────────────────
    votes    = z_result + db_result + iso_result
    ensemble = np.where(votes >= 2, 1, 0)

    # Ensemble continuous score: min-max normalise each algorithm's score
    # to [0,1] so no single algorithm's raw scale dominates, then average.
    # All three component scores are already oriented so higher = more
    # anomalous, so the average preserves that orientation. ✓
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