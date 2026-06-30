import numpy as np
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


def _sanitise(arr):
    """Replace any NaN/Inf with finite fallback values so downstream
    sklearn metrics (roc_curve, precision_recall_curve) never crash on
    assert_all_finite. NaN -> 0.0, +Inf -> max finite value in arr,
    -Inf -> min finite value in arr (falls back to 0.0 if everything
    is non-finite)."""
    arr = np.asarray(arr, dtype=float)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr)
    finite_vals = arr[finite_mask]
    fmax = finite_vals.max()
    fmin = finite_vals.min()
    arr = np.where(np.isnan(arr), 0.0, arr)
    arr = np.where(np.isposinf(arr), fmax, arr)
    arr = np.where(np.isneginf(arr), fmin, arr)
    return arr


def detect(X, z_threshold, eps, min_samples, contamination):

    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    #################################
    # Z-score
    #################################
    # zscore() divides by std-dev per column; a zero-variance column
    # produces NaN for every row in that column — guard against that
    # before taking abs()/max().
    z_raw = zscore(X, axis=0)
    z_raw = np.nan_to_num(z_raw, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.abs(z_raw)

    z_score_continuous = _sanitise(z.max(axis=1))
    z_result = (z > z_threshold).any(axis=1).astype(int)

    #################################
    # DBSCAN
    #################################
    db = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )
    labels = db.fit_predict(X)
    db_result = np.where(labels == -1, 1, 0)

    # Continuous score: mean distance to nearest neighbours. Guard the
    # neighbour count against having more neighbours requested than
    # points available (which raises in NearestNeighbors, not just
    # returns NaN), and guard the final array against any residual
    # non-finite values before returning it.
    k = max(1, min(min_samples, n_samples - 1))
    if n_samples > 1:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        db_score_continuous = distances[:, 1:].mean(axis=1)  # exclude self (distance 0)
    else:
        db_score_continuous = np.zeros(n_samples)
    db_score_continuous = _sanitise(db_score_continuous)

    #################################
    # Isolation Forest
    #################################
    iso = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    iso.fit(X)
    iso_result = np.where(
        iso.predict(X) == -1,
        1,
        0
    )

    iso_score_continuous = _sanitise(-iso.decision_function(X))

    #################################
    # Ensemble
    #################################
    votes = z_result + db_result + iso_result
    ensemble = np.where(votes >= 2, 1, 0)

    def _minmax(a):
        a = _sanitise(a)
        lo, hi = a.min(), a.max()
        if hi - lo < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    ensemble_score_continuous = _sanitise(
        (
            _minmax(z_score_continuous)
            + _minmax(db_score_continuous)
            + _minmax(iso_score_continuous)
        ) / 3.0
    )

    continuous_scores = {
        "Z-Score":          z_score_continuous,
        "DBSCAN":           db_score_continuous,
        "Isolation Forest": iso_score_continuous,
        "Ensemble":         ensemble_score_continuous,
    }

    return z_result, db_result, iso_result, votes, ensemble, continuous_scores