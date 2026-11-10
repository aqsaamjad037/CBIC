
import os
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA = os.environ.get("IBD_DIR", os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("IBD_OUT", DATA)


def P(name):
    for n in (name, name.replace(".csv.gz", "_csv.gz")):
        p = os.path.join(DATA, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def make_model():
    return Pipeline([("select", SelectKBest(f_classif, k=500)),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000, random_state=42))])


Z = pd.read_csv(P("combined5_zscore_expression.csv.gz"), index_col=0)
meta = pd.read_csv(P("combined5_metadata.csv"))
assert list(Z.columns) == list(meta["Sample"]), "sample order differs between matrix and metadata"
X, g = Z.T.values, meta["Group"].values

m1 = make_model().fit(X, (g != "Control").astype(int))
cu = g != "Control"
m2 = make_model().fit(X[cu], (g[cu] == "CD").astype(int))

genes = list(Z.index)
joblib.dump({"level1": m1, "level2": m2, "genes": genes}, os.path.join(OUT, "CBIC_final_models.joblib"))
print("saved CBIC_final_models.joblib | genes: %d | samples: %d (Level 2: %d)" % (len(genes), X.shape[0], cu.sum()))
