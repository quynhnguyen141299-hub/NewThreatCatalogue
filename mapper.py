# mapper.py
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, os, ast
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

STRIDE_TO_TECHNIQUES = {
    "Spoofing":             ["T1078", "T1190"],
    "Tampering":            ["T1565", "T1203"],
    "InformationDisclosure":["T1213", "T1078"],
    "DenialOfService":      ["T1498", "T1499"],
    "Repudiation":          ["T1070", "T1562"],
    "ElevationOfPrivilege": ["T1068", "T1078"],
}
LAYER_TO_TECHNIQUES = {
    "Access":   ["T1190", "T1078"],
    "Service":  ["T1203", "T1565"],
    "Asset":    ["T1657", "T1565"],
    "Platform": ["T1068", "T1562"],
}
ALL_TECHNIQUES = sorted(set(
    t for ts in list(STRIDE_TO_TECHNIQUES.values()) +
                list(LAYER_TO_TECHNIQUES.values()) for t in ts
))

class NISTMapper:

    def __init__(self, mappings_path="data/attack_to_nist.csv"):
        self.mappings_path = mappings_path
        self.clf     = None
        self.mlb     = MultiLabelBinarizer()
        self.trained = False

    def _build_feature_vector(self, stride_tags, asap_layer):
        techniques = []
        for tag in stride_tags:
            techniques += STRIDE_TO_TECHNIQUES.get(tag.strip(), [])
        techniques += LAYER_TO_TECHNIQUES.get(asap_layer, [])
        vec = np.zeros(len(ALL_TECHNIQUES))
        for t in techniques:
            if t in ALL_TECHNIQUES:
                vec[ALL_TECHNIQUES.index(t)] = 1
        return vec

    def train(self):
        if not os.path.exists(self.mappings_path):
            print("Mappings file not found. Using rule-based fallback.")
            return False
        df = pd.read_csv(self.mappings_path)
        # Expected columns: technique_id, control_id
        if "technique_id" not in df.columns or "control_id" not in df.columns:
            print("Unexpected CSV format.")
            return False
        grouped = df.groupby("technique_id")["control_id"].apply(list).reset_index()
        X, y_raw = [], []
        for _, row in grouped.iterrows():
            vec = np.zeros(len(ALL_TECHNIQUES))
            if row["technique_id"] in ALL_TECHNIQUES:
                vec[ALL_TECHNIQUES.index(row["technique_id"])] = 1
            X.append(vec)
            y_raw.append(row["control_id"])
        if len(X) < 5:
            return False
        y = self.mlb.fit_transform(y_raw)
        X = np.array(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.clf = OneVsRestClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        )
        self.clf.fit(X_train, y_train)
        self.trained = True
        print("Classifier trained.")
        print(classification_report(
            y_test, self.clf.predict(X_test),
            target_names=self.mlb.classes_, zero_division=0
        ))
        return True

    def predict(self, stride_tags, asap_layer, threshold=0.3):
        if not self.trained:
            return {"controls": [], "confidence": {}, "low_confidence": True, "source": "fallback"}
        vec  = self._build_feature_vector(stride_tags, asap_layer)
        prob = self.clf.predict_proba([vec])
        # OneVsRestClassifier returns a 2D array directly
        if hasattr(prob, 'shape'):
            # Single array of shape (1, n_classes)
            confidence = {
                self.mlb.classes_[i]: float(prob[0][i])
                for i in range(len(self.mlb.classes_))
            }
        else:
            # List of arrays, one per class
            confidence = {
                self.mlb.classes_[i]: float(prob[i][0][1])
                for i in range(len(self.mlb.classes_))
                if hasattr(prob[i][0], '__len__') and len(prob[i][0]) > 1
            }
        controls       = [c for c, p in confidence.items() if p >= threshold]
        max_conf       = max(confidence.values()) if confidence else 0.0
        low_confidence = max_conf < threshold
        return {
            "controls":       sorted(controls),
            "confidence":     confidence,
            "low_confidence": low_confidence,
            "source":         "ml",
        }

    def save(self, path="data/nist_mapper.joblib"):
        joblib.dump({"clf": self.clf, "mlb": self.mlb, "trained": self.trained}, path)

    def load(self, path="data/nist_mapper.joblib"):
        if os.path.exists(path):
            data         = joblib.load(path)
            self.clf     = data["clf"]
            self.mlb     = data["mlb"]
            self.trained = data["trained"]
            return True
        return False