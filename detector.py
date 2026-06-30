import numpy as np
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


def detect(X, z_threshold, eps, min_samples, contamination):

    #################################
    # Z-score
    #################################
    z = np.abs(zscore(X))
    # Continuous score: the single largest |z| across all features for
    # each row. This is what actually drove the binary decision, so it
    # is a meaningful continuous signal for ROC/PR curves (higher = more
    # anomalous), unlike the thresholded 0/1 result.
    z_score_continuous = z.max(axis=1)
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

    # Continuous score: distance to the nearest neighbour. DBSCAN itself
    # has no native anomaly score, so we derive one — points far from
    # their nearest neighbour are the ones DBSCAN tends to call noise
    # (-1), so this distance is a reasonable continuous proxy (higher =
    # more anomalous) that still respects DBSCAN's own geometry.
    k = max(1, min(min_samples, len(X) - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    db_score_continuous = distances[:, 1:].mean(axis=1)  # exclude self (distance 0)

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

    # Continuous score: Isolation Forest's own decision_function returns
    # higher values for normal points and lower (more negative) values
    # for anomalies, so we flip the sign (higher = more anomalous) to
    # match the convention used by the other two scores above.
    iso_score_continuous = -iso.decision_function(X)

    #################################
    # Ensemble
    #################################
    votes = z_result + db_result + iso_result
    ensemble = np.where(votes >= 2, 1, 0)

    # Continuous ensemble score: simple average of the three normalised
    # continuous scores (each min-max scaled to [0, 1] first so no single
    # algorithm's raw scale dominates the average).
    def _minmax(a):
        a = np.asarray(a, dtype=float)
        lo, hi = a.min(), a.max()
        if hi - lo < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    ensemble_score_continuous = (
        _minmax(z_score_continuous)
        + _minmax(db_score_continuous)
        + _minmax(iso_score_continuous)
    ) / 3.0

    continuous_scores = {
        "Z-Score":          z_score_continuous,
        "DBSCAN":           db_score_continuous,
        "Isolation Forest": iso_score_continuous,
        "Ensemble":         ensemble_score_continuous,
    }

    return z_result, db_result, iso_result, votes, ensemble, continuous_scores
