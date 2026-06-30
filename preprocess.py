import pandas as pd
import json

from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df):

    def parse_json(x):
        try:
            return json.loads(x)
        except:
            return {}

    details = df["details"].apply(parse_json)
    details_df = pd.json_normalize(details)

    df = pd.concat(
        [df.drop(columns=["details"]), details_df],
        axis=1
    )

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
        df["hour"] = df["ts"].dt.hour
        df["minute"] = df["ts"].dt.minute
        df["second"] = df["ts"].dt.second
        df.drop(columns=["ts"], inplace=True)

    # Extract ground-truth label BEFORE it touches the feature matrix.
    # "label" comes from the simulator (0 = benign, 1 = attacker-driven).
    # It must never be used as a model feature (that would be leaking the
    # answer into the detector), so pull it out into its own series and
    # drop it from the dataframe that becomes X.
    if "label" in df.columns:
        ground_truth_label = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        df = df.drop(columns=["label"])
    else:
        ground_truth_label = pd.Series([0] * len(df), index=df.index)

    # Fill NaN values
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype).startswith('string'):
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = df[col].fillna(0)

    # Encode ALL non-numeric columns
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))

    # Ensure all columns are float
    df = df.astype(float)

    # Re-attach the ground-truth label to the processed dataframe for
    # display/evaluation purposes ONLY (it is NOT part of feature_names
    # and is NOT included in X below).
    df["ground_truth_label"] = ground_truth_label.values

    # Save feature names before scaling — deliberately excludes
    # ground_truth_label so it can never leak into the detection models.
    feature_names = [c for c in df.columns if c != "ground_truth_label"]

    # Scale only the real features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_names])

    return X, df, feature_names, scaler