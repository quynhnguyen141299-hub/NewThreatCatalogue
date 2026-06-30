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
        df["ts"] = pd.to_datetime(df["ts"])
        df["hour"] = df["ts"].dt.hour
        df["minute"] = df["ts"].dt.minute
        df["second"] = df["ts"].dt.second
        df.drop(columns=["ts"], inplace=True)

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

    # Save feature names before scaling
    feature_names = df.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    return X, df, feature_names, scaler