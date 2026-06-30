import numpy as np
from scipy.stats import zscore

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest


def detect(X, z_threshold, eps, min_samples, contamination):

    #################################
    # Z-score
    #################################

    z = np.abs(zscore(X))

    z_result = (z > z_threshold).any(axis=1).astype(int)

    #################################
    # DBSCAN
    #################################

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = db.fit_predict(X)

    db_result = np.where(labels == -1,1,0)

    #################################
    # Isolation Forest
    #################################

    iso = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    iso.fit(X)

    iso_result = np.where(
        iso.predict(X)==-1,
        1,
        0
    )

    #################################
    # Ensemble
    #################################

    votes = z_result + db_result + iso_result

    ensemble = np.where(votes>=2,1,0)

    return z_result, db_result, iso_result, votes, ensemble