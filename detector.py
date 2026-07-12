import numpy as np
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


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


def detect(
    X,
    z_threshold,
    eps,
    min_samples,
    contamination,
    scale_for_dbscan=True,
    use_lof_score=True,
    random_state=42,
    n_jobs=-1,
    nn_algorithm="auto",
    vote_threshold=1,
    detector_weights=None,
):
    """
    vote_threshold: minimum number of detectors that must fire for a
        point to be flagged in `ensemble`. Default is 1 ("any detector
        fires"), which maximises recall — appropriate for threat
        detection, where a missed threat (false negative) is typically
        far costlier than an extra alert a human triages (false
        positive). Set to 2 or 3 for a stricter, higher-precision,
        lower-recall ensemble.
    detector_weights: optional dict like {"Z-Score": 1.0, "DBSCAN": 1.5,
        "Isolation Forest": 1.0} to weight `ensemble_score_continuous`
        toward whichever detector catches your threats best historically.
        Defaults to equal weighting.
    """
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    #################################
    # Z-score
    #################################
    z_raw = zscore(X, axis=0)
    z_raw = np.nan_to_num(z_raw, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.abs(z_raw)
    z_score_continuous = _sanitise(z.max(axis=1))
    z_result = (z > z_threshold).any(axis=1).astype(int)

    #################################
    # DBSCAN
    #################################
    # CHANGE: DBSCAN's eps is a raw Euclidean distance threshold. If
    # input features are on different scales, large-magnitude columns
    # dominate distance and eps becomes meaningless. Z-score and
    # Isolation Forest don't have this problem (the former standardises
    # internally, the latter splits per-feature), so DBSCAN alone needs
    # scaling before it's comparable to the other two detectors.
    X_db = StandardScaler().fit_transform(X) if scale_for_dbscan else X

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_db)
    db_result = np.where(labels == -1, 1, 0)

    k = max(1, min(min_samples, n_samples - 1))
    if n_samples > 1:
        if use_lof_score:
            # CHANGE: plain mean k-NN distance treats "sparse but locally
            # uniform" regions the same as "sparse because of an outlier
            # next to a dense cluster" — it ignores local density
            # variation. LOF corrects for this by comparing a point's
            # density to its neighbours' density, which tends to rank
            # points closer to how a human-labelled anomaly set expects.
            lof = LocalOutlierFactor(n_neighbors=k + 1)
            lof.fit_predict(X_db)
            db_score_continuous = -lof.negative_outlier_factor_
        else:
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_db)
            distances, _ = nbrs.kneighbors(X_db)
            db_score_continuous = distances[:, 1:].mean(axis=1)
    else:
        db_score_continuous = np.zeros(n_samples)
    db_score_continuous = _sanitise(db_score_continuous)

    #################################
    # Isolation Forest
    #################################
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X)
    iso_result = np.where(iso.predict(X) == -1, 1, 0)
    iso_score_continuous = _sanitise(-iso.decision_function(X))

    #################################
    # Align flagging rates across detectors
    #################################
    # CHANGE: z_threshold and eps/min_samples are picked independently
    # of `contamination`, so each detector can end up flagging a wildly
    # different fraction of points (e.g. Isolation Forest flags exactly
    # `contamination`%, DBSCAN might flag 40% as noise, z-score might
    # flag 1%). That mismatch alone can wreck precision/recall even when
    # each detector's *ranking* of points is reasonable. Re-deriving
    # discrete flags from each continuous score at the same target rate
    # makes the three detectors comparable and the majority vote meaningful.
    def _flag_at_rate(score, rate):
        if rate <= 0:
            return np.zeros_like(score, dtype=int)
        cutoff = np.quantile(score, 1 - rate)
        return (score >= cutoff).astype(int)

    z_result_aligned = _flag_at_rate(z_score_continuous, contamination)
    db_result_aligned = _flag_at_rate(db_score_continuous, contamination)
    iso_result_aligned = iso_result  # already at `contamination` rate by construction

    #################################
    # Ensemble
    #################################
    # CHANGE: default vote_threshold=1 means ANY single detector firing
    # is enough to flag a point ("OR" logic), not majority ("AND"-ish,
    # 2-of-3) as before. Different detectors catch different threat
    # shapes — z-score catches extreme single-feature spikes, DBSCAN/LOF
    # catches density-based outliers, Isolation Forest catches
    # multivariate splits. Requiring agreement throws away threats only
    # one method can see. This trades precision for recall, which is
    # normally the right trade for threat detection (missed threat >>
    # extra alert to triage). Raise vote_threshold to 2 or 3 to go back
    # toward higher precision / lower recall.
    votes = z_result_aligned + db_result_aligned + iso_result_aligned
    ensemble = np.where(votes >= vote_threshold, 1, 0)

    def _minmax(a):
        a = _sanitise(a)
        lo, hi = a.min(), a.max()
        if hi - lo < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    # CHANGE: optional per-detector weighting of the continuous ensemble
    # score, so if e.g. Isolation Forest has historically caught more of
    # your real threats, you can weight it up without discarding the
    # other two detectors entirely.
    weights = detector_weights or {"Z-Score": 1.0, "DBSCAN": 1.0, "Isolation Forest": 1.0}
    total_weight = sum(weights.values())
    ensemble_score_continuous = _sanitise(
        (
            weights.get("Z-Score", 1.0) * _minmax(z_score_continuous)
            + weights.get("DBSCAN", 1.0) * _minmax(db_score_continuous)
            + weights.get("Isolation Forest", 1.0) * _minmax(iso_score_continuous)
        )
        / total_weight
    )

    continuous_scores = {
        "Z-Score": z_score_continuous,
        "DBSCAN": db_score_continuous,
        "Isolation Forest": iso_score_continuous,
        "Ensemble": ensemble_score_continuous,
    }

    # NOTE: original (un-aligned, threshold-driven) discrete results are
    # still returned too, in case you want to compare "raw threshold"
    # vs "rate-aligned" behaviour.
    return (
        z_result,
        db_result,
        iso_result,
        votes,
        ensemble,
        continuous_scores,
    )


def flag_at_target_recall(scores, y_true, target_recall=0.95):
    """
    Pick a threshold on a continuous score (e.g. continuous_scores["Ensemble"])
    that achieves at least `target_recall` on known labels, and report the
    precision that results at that threshold. This is generally a better way
    to set an operating point for threat detection than a fixed vote count or
    contamination guess, because it directly targets the metric you actually
    care about (missed-threat rate) instead of an indirect proxy.

    Requires labelled data (y_true), e.g. from a held-out validation set or
    known historical incidents. Returns (threshold, flags, achieved_precision,
    achieved_recall).
    """
    from sklearn.metrics import precision_recall_curve

    scores = _sanitise(np.asarray(scores, dtype=float))
    y_true = np.asarray(y_true)

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns arrays 1 longer than `thresholds`;
    # thresholds[i] corresponds to precision[i]/recall[i].
    valid = np.where(recall[:-1] >= target_recall)[0]
    if len(valid) == 0:
        # Can't hit target recall even at the lowest threshold; flag everything.
        chosen_idx = 0
        chosen_threshold = scores.min()
    else:
        # Among thresholds achieving target recall, pick the one with
        # highest precision (least false-positive cost).
        chosen_idx = valid[np.argmax(precision[valid])]
        chosen_threshold = thresholds[chosen_idx] if chosen_idx < len(thresholds) else scores.min()

    flags = (scores >= chosen_threshold).astype(int)
    achieved_precision = precision[chosen_idx]
    achieved_recall = recall[chosen_idx]
    return chosen_threshold, flags, achieved_precision, achieved_recall


def max_recall_ensemble(continuous_scores, y_true, target_recall=0.95):
    """
    Push recall as far as this ensemble can go: instead of aligning all
    three detectors to one shared flagging rate and voting (which caps
    recall at whatever the *combined* vote allows), tune EACH detector's
    threshold independently to hit `target_recall` on its own, then take
    the UNION of all three resulting flag sets.

    Why this is the ceiling for this ensemble: a threat missed by
    Isolation Forest but caught by DBSCAN/LOF is preserved here, and vice
    versa — nothing is thrown away by requiring agreement. The union's
    recall is mathematically >= the best individual detector's recall,
    and typically higher, since different detectors miss different
    threats. The cost is precision: you will get more false positives
    than any single detector alone. That is the correct trade for threat
    detection (a human triaging extra alerts is far cheaper than a
    missed intrusion).

    Requires labelled data (y_true) — even a small validated slice or
    known past incidents is enough; this cannot be done label-free,
    since "maximum recall" is only measurable against ground truth.

    Returns: dict with per-detector thresholds/flags, the unioned flags,
    and the achieved recall/precision of the union.
    """
    from sklearn.metrics import precision_score, recall_score

    y_true = np.asarray(y_true)
    per_detector = {}
    union_flags = np.zeros(len(y_true), dtype=int)

    for name in ("Z-Score", "DBSCAN", "Isolation Forest"):
        threshold, flags, prec, rec = flag_at_target_recall(
            continuous_scores[name], y_true, target_recall=target_recall
        )
        per_detector[name] = {
            "threshold": threshold,
            "flags": flags,
            "precision": prec,
            "recall": rec,
        }
        union_flags = np.logical_or(union_flags, flags).astype(int)

    result = {
        "per_detector": per_detector,
        "union_flags": union_flags,
        "union_recall": recall_score(y_true, union_flags),
        "union_precision": precision_score(y_true, union_flags, zero_division=0),
    }
    return result


def tune_detectors(
    X,
    y_true,
    iso_param_grid=None,
    min_samples_grid=None,
    eps_grid=None,
    beta=2.0,
    random_state=42,
):
    """
    Grid-search each detector's hyperparameters against labelled data.

    Objective: F-beta with beta > 1 (default beta=2), which weights
    recall roughly 4x as heavily as precision in the score -- matching
    the threat-detection priority of "missed threat >> extra alert".
    Isolation Forest's ranking quality is scored by PR-AUC directly
    (average_precision_score), since PR-AUC IS the area under the
    precision/recall trade-off curve and is threshold-independent.

    What's tuned and why:
      - Isolation Forest: n_estimators, max_samples, max_features.
        NOTE: `contamination` is deliberately excluded here -- it only
        shifts the predict() decision threshold (offset_), it does not
        change decision_function's ranking of points, so it has zero
        effect on PR-AUC. Set contamination afterwards from your target
        recall instead (via flag_at_target_recall), not by grid search.
      - DBSCAN/LOF: min_samples controls LOF's neighbourhood size,
        which DOES change the ranking quality of the continuous score
        -- tuned by PR-AUC. eps only affects DBSCAN's own discrete noise
        flag (db_result), which is independent of the LOF score, so
        it's tuned separately by F-beta on the resulting binary flags.
      - Z-score: NOT tuned. z_threshold only changes the discrete flag,
        not the underlying ranking (max abs z-score per row) -- there
        is nothing here for a grid search to improve. Use
        flag_at_target_recall on z_score_continuous directly instead.

    Returns a dict of best params + achieved scores per detector. Feed
    the results into `detect()` and/or `max_recall_ensemble()`.
    """
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import average_precision_score, fbeta_score

    y_true = np.asarray(y_true)
    n_samples = X.shape[0]

    # ---- Isolation Forest: tune ranking quality (PR-AUC) ----
    if iso_param_grid is None:
        iso_param_grid = {
            "n_estimators": [100, 200, 300],
            "max_samples": ["auto", 0.5, 0.8],
            "max_features": [1.0, 0.8],
        }
    best_iso = {"pr_auc": -1, "params": None}
    for params in ParameterGrid(iso_param_grid):
        iso = IsolationForest(random_state=random_state, **params)
        iso.fit(X)
        score = _sanitise(-iso.decision_function(X))
        ap = average_precision_score(y_true, score)
        if ap > best_iso["pr_auc"]:
            best_iso = {"pr_auc": ap, "params": params}

    # ---- LOF: tune neighbourhood size (PR-AUC) ----
    if min_samples_grid is None:
        min_samples_grid = [3, 5, 10, 15, 20]
    X_db = StandardScaler().fit_transform(X)

    best_lof = {"pr_auc": -1, "min_samples": None}
    for m in min_samples_grid:
        k = max(1, min(m, n_samples - 1))
        lof = LocalOutlierFactor(n_neighbors=k + 1)
        lof.fit_predict(X_db)
        score = _sanitise(-lof.negative_outlier_factor_)
        ap = average_precision_score(y_true, score)
        if ap > best_lof["pr_auc"]:
            best_lof = {"pr_auc": ap, "min_samples": m}

    # ---- DBSCAN: tune discrete noise-flag quality (F-beta, recall-weighted) ----
    if eps_grid is None:
        std = X_db.std()  # X_db is already standardised, std ~= 1 per column on average
        eps_grid = [std * m for m in (0.3, 0.5, 0.8, 1.2, 1.6, 2.0)]

    best_dbscan = {"f_beta": -1, "eps": None, "min_samples": None}
    for m in min_samples_grid:
        for eps in eps_grid:
            labels = DBSCAN(eps=eps, min_samples=m).fit_predict(X_db)
            flags = np.where(labels == -1, 1, 0)
            fb = fbeta_score(y_true, flags, beta=beta, zero_division=0)
            if fb > best_dbscan["f_beta"]:
                best_dbscan = {"f_beta": fb, "eps": eps, "min_samples": m}

    return {
        "isolation_forest": best_iso,
        "lof": best_lof,
        "dbscan_flag": best_dbscan,
    }


def detect_tuned(X, y_true, target_recall=0.95, beta=2.0, random_state=42):
    """
    Convenience wrapper: tune every detector against y_true, rebuild the
    continuous scores with the tuned hyperparameters, then run
    max_recall_ensemble on the tuned scores. This is the full pipeline --
    tuning + union -- rather than tuning alone.

    Requires labelled data. Returns (tuning_results, continuous_scores,
    max_recall_result).
    """
    tuning = tune_detectors(X, y_true, beta=beta, random_state=random_state)
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    # Z-score: unchanged, nothing to tune.
    z_raw = np.nan_to_num(zscore(X, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    z_score_continuous = _sanitise(np.abs(z_raw).max(axis=1))

    # Isolation Forest with tuned params.
    iso = IsolationForest(random_state=random_state, **tuning["isolation_forest"]["params"])
    iso.fit(X)
    iso_score_continuous = _sanitise(-iso.decision_function(X))

    # LOF with tuned min_samples.
    X_db = StandardScaler().fit_transform(X)
    m = tuning["lof"]["min_samples"]
    k = max(1, min(m, n_samples - 1))
    lof = LocalOutlierFactor(n_neighbors=k + 1)
    lof.fit_predict(X_db)
    db_score_continuous = _sanitise(-lof.negative_outlier_factor_)

    continuous_scores = {
        "Z-Score": z_score_continuous,
        "DBSCAN": db_score_continuous,
        "Isolation Forest": iso_score_continuous,
    }

    max_recall_result = max_recall_ensemble(continuous_scores, y_true, target_recall=target_recall)
    return tuning, continuous_scores, max_recall_result


if __name__ == "__main__":
    # Minimal runnable example / smoke test.
    rng = np.random.RandomState(0)
    normal = rng.normal(loc=0, scale=1, size=(200, 3))
    anomalies = rng.uniform(low=6, high=10, size=(10, 3))
    X_demo = np.vstack([normal, anomalies])
    y_true = np.array([0] * 200 + [1] * 10)

    # vote_threshold=1 (default): any single detector firing is enough —
    # maximises recall, the usual priority for threat detection.
    z_res, db_res, iso_res, votes, ensemble, scores = detect(
        X_demo,
        z_threshold=3.0,
        eps=0.8,
        min_samples=5,
        contamination=10 / 210,
    )

    print("=== vote_threshold=1 (any detector fires) ===")
    print(f"Ensemble caught {ensemble[-10:].sum()} / 10 true anomalies")
    print(f"False positives among the 200 normals: {ensemble[:200].sum()}")

    try:
        from sklearn.metrics import (
            classification_report,
            roc_auc_score,
            average_precision_score,
        )

        print(classification_report(y_true, ensemble))
        print(f"ROC-AUC (ensemble score):  {roc_auc_score(y_true, scores['Ensemble']):.3f}")
        print(f"PR-AUC  (ensemble score):  {average_precision_score(y_true, scores['Ensemble']):.3f}")
    except ImportError:
        pass

    # If you have labels (even from a small validation slice or known past
    # incidents), pick the operating point directly by target recall instead
    # of guessing at vote_threshold/contamination.
    threshold, flags, prec, rec = flag_at_target_recall(
        scores["Ensemble"], y_true, target_recall=0.95
    )
    print(f"\n=== Recall-targeted threshold (target_recall=0.95) ===")
    print(f"Chosen threshold: {threshold:.4f}")
    print(f"Achieved precision: {prec:.3f}, achieved recall: {rec:.3f}")
    print(f"Flagged {flags.sum()} of {len(flags)} points")

    # Furthest push: independently maximise each detector's recall, then
    # union the results. This is the highest recall this ensemble can
    # achieve given labels — see max_recall_ensemble's docstring.
    max_recall_result = max_recall_ensemble(scores, y_true, target_recall=0.95)
    print(f"\n=== Max-recall union ensemble (target_recall=0.95 per detector) ===")
    for name, info in max_recall_result["per_detector"].items():
        print(f"  {name}: recall={info['recall']:.3f}, precision={info['precision']:.3f}")
    print(f"UNION recall:    {max_recall_result['union_recall']:.3f}")
    print(f"UNION precision: {max_recall_result['union_precision']:.3f}")
    print(f"UNION flagged {max_recall_result['union_flags'].sum()} of {len(y_true)} points")

    # Furthest-furthest push: tune each detector's hyperparameters
    # against labels first, THEN take the max-recall union of the
    # tuned scores. Compare this to the untuned union above.
    print(f"\n=== Tuned pipeline (detect_tuned) ===")
    tuning, tuned_scores, tuned_result = detect_tuned(X_demo, y_true, target_recall=0.95)
    print("Best hyperparameters found:")
    print(f"  Isolation Forest: {tuning['isolation_forest']['params']} "
          f"(PR-AUC={tuning['isolation_forest']['pr_auc']:.3f})")
    print(f"  LOF min_samples:  {tuning['lof']['min_samples']} "
          f"(PR-AUC={tuning['lof']['pr_auc']:.3f})")
    print(f"  DBSCAN eps/min_samples: {tuning['dbscan_flag']['eps']:.3f} / "
          f"{tuning['dbscan_flag']['min_samples']} (F-beta={tuning['dbscan_flag']['f_beta']:.3f})")
    print("\nPer-detector recall/precision at target_recall=0.95, AFTER tuning:")
    for name, info in tuned_result["per_detector"].items():
        print(f"  {name}: recall={info['recall']:.3f}, precision={info['precision']:.3f}")
    print(f"TUNED UNION recall:    {tuned_result['union_recall']:.3f}")
    print(f"TUNED UNION precision: {tuned_result['union_precision']:.3f}")
    print(f"TUNED UNION flagged {tuned_result['union_flags'].sum()} of {len(y_true)} points")
