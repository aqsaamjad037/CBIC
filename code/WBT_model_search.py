
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FOLDER = r"CBIC"

SEED, N_BOOT = 42, 1000
TRAIN, TESTS = "GSE186507", ["GSE112057", "GSE119600"]
K_LIST = [100, 300, 1000]
MODELS = [("ridge C=0.003", "lr2", 0.003), ("ridge C=0.01", "lr2", 0.01), ("ridge C=0.1", "lr2", 0.1),
          ("lasso C=0.05", "lr1", 0.05), ("lasso C=0.2", "lr1", 0.2), ("random forest", "rf", None)]
CONFIGS = [(k, m) for k in K_LIST for m in MODELS]
BASELINE = (500, ("ridge C=0.01", "lr2", 0.01))
PAPER = {("CD vs Control", "GSE112057"): 0.90, ("CD vs Control", "GSE119600"): 0.84,
         ("UC vs Control", "GSE112057"): 0.94, ("UC vs Control", "GSE119600"): 0.74}

Z = pd.read_csv(os.path.join(FOLDER, "combined_zscore_expression.csv.gz"), index_col=0)
meta = pd.read_csv(os.path.join(FOLDER, "combined_metadata.csv"))
assert list(Z.columns) == list(meta["Sample"])
X_ALL, grp, dsn = Z.T.values, meta["Group"].values, meta["Dataset"].values
rng = np.random.RandomState(SEED)
print("Datasets used (Li et al. 2026 only):", sorted(set(dsn)))


def fit_predict(Xtr, ytr, Xte, k, model):
    _, kind, c = model
    sel = SelectKBest(f_classif, k=min(k, Xtr.shape[1])).fit(Xtr, ytr)
    Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", min_samples_leaf=3,
                                     n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=c, penalty="l2" if kind == "lr2" else "l1", solver="liblinear",
                             class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def boot_ci(yt, p):
    a = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(yt), len(yt))
        if len(set(yt[i])) == 2:
            a.append(roc_auc_score(yt[i], p[i]))
    return np.percentile(a, [2.5, 97.5])


TASKS = {"CD vs Control": (grp != "UC", (grp == "CD").astype(int)),
         "UC vs Control": (grp != "CD", (grp == "UC").astype(int)),
         "IBD (CD+UC) vs Control": (np.ones(len(grp), bool), (grp != "Control").astype(int))}
res, cvrows = [], []
for task, (keep, y) in TASKS.items():
    tr = np.where(keep & (dsn == TRAIN))[0]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv = {}
    for cfg in CONFIGS:
        a = []
        for a_i, b_i in skf.split(tr, y[tr]):
            p = fit_predict(X_ALL[tr[a_i]], y[tr[a_i]], X_ALL[tr[b_i]], cfg[0], cfg[1])
            a.append(roc_auc_score(y[tr[b_i]], p))
        cv[cfg] = float(np.mean(a))
        cvrows.append({"Task": task, "Setting": "K=%d, %s" % (cfg[0], cfg[1][0]), "CV_AUC_on_GSE186507": cv[cfg]})
    best = max(cv, key=cv.get)
    print("\n%s: chosen by CV on GSE186507 -> K=%d, %s (CV AUC %.3f)" % (task, best[0], best[1][0], cv[best]))
    for test in TESTS:
        te = np.where(keep & (dsn == test))[0]
        for label, cfg in (("CHOSEN (CV on train only)", best), ("BASELINE (Step 30)", BASELINE)):
            p = fit_predict(X_ALL[tr], y[tr], X_ALL[te], cfg[0], cfg[1])
            lo, hi = boot_ci(y[te], p)
            res.append({"Task": task, "Test_dataset": test, "Method": label,
                        "Setting": "K=%d, %s" % (cfg[0], cfg[1][0]),
                        "AUC": roc_auc_score(y[te], p), "CI_low": lo, "CI_high": hi,
                        "BalancedAcc": balanced_accuracy_score(y[te], (p > .5).astype(int)),
                        "Paper_AUC": PAPER.get((task, test), np.nan), "n_test": len(te)})

R = pd.DataFrame(res)
R.to_csv(os.path.join(FOLDER, "li2026_model_search_results.csv"), index=False)
pd.DataFrame(cvrows).to_csv(os.path.join(FOLDER, "li2026_model_search_cv_table.csv"), index=False)
R["AUC_95CI"] = ["%.3f [%.3f-%.3f]" % (a, l, h) for a, l, h in zip(R["AUC"], R["CI_low"], R["CI_high"])]
