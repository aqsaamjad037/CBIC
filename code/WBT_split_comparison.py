
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FOLDER = r"CBIC"

SEED, N_BOOT, K = 42, 1000, 500
TRAIN_P = "GSE186507"
TESTS_P = ["GSE112057", "GSE119600"]

Z = pd.read_csv(os.path.join(FOLDER, "combined_zscore_expression.csv.gz"), index_col=0)
meta = pd.read_csv(os.path.join(FOLDER, "combined_metadata.csv"))
assert list(Z.columns) == list(meta["Sample"]), "Sample order differs between expression and metadata"
print("Samples per dataset and group (ours):")
print(pd.crosstab(meta["Dataset"], meta["Group"]).to_string())
print("(Paper Table 1: GSE186507 CD 429 / UC 378 / Control 209; GSE112057 CD 60 / UC 15 / Control 12; "
      "GSE119600 CD 95 / UC 93 / Control 47)\n")

X_all = Z.T.values
grp = meta["Group"].values
dsn = meta["Dataset"].values
rng = np.random.RandomState(SEED)

TASKS = {
    "CD vs Control": (lambda g: g != "UC", lambda g: g == "CD"),
    "UC vs Control": (lambda g: g != "CD", lambda g: g == "UC"),
    "IBD (CD+UC) vs Control": (lambda g: np.ones(len(g), bool), lambda g: g != "Control"),
}


def boot_ci(yt, p):
    a = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(yt), len(yt))
        if len(set(yt[i])) == 2:
            a.append(roc_auc_score(yt[i], p[i]))
    return np.percentile(a, [2.5, 97.5])


def fit_predict(tr_idx, te_idx, y_all):
    Xtr, Xte = X_all[tr_idx], X_all[te_idx]
    sel = SelectKBest(f_classif, k=K).fit(Xtr, y_all[tr_idx])
    Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(Xtr), y_all[tr_idx])
    return clf.predict_proba(sc.transform(Xte))[:, 1]


rows = []
for task, (keep_fn, pos_fn) in TASKS.items():
    keep = keep_fn(grp)
    y_all = pos_fn(grp).astype(int)
    # Setting P: paper split
    for test in TESTS_P:
        tr = np.where(keep & (dsn == TRAIN_P))[0]
        te = np.where(keep & (dsn == test))[0]
        p = fit_predict(tr, te, y_all)
        lo, hi = boot_ci(y_all[te], p)
        rows.append({"Setting": "P: train GSE186507 only", "Task": task, "Test_dataset": test,
                     "AUC": roc_auc_score(y_all[te], p), "CI_low": lo, "CI_high": hi,
                     "BalancedAcc": balanced_accuracy_score(y_all[te], (p > .5).astype(int)),
                     "n_train": len(tr), "n_test": len(te)})
    for test in list(dict.fromkeys(dsn)):
        tr = np.where(keep & (dsn != test))[0]
        te = np.where(keep & (dsn == test))[0]
        p = fit_predict(tr, te, y_all)
        lo, hi = boot_ci(y_all[te], p)
        rows.append({"Setting": "L: LODO (train on other 2)", "Task": task, "Test_dataset": test,
                     "AUC": roc_auc_score(y_all[te], p), "CI_low": lo, "CI_high": hi,
                     "BalancedAcc": balanced_accuracy_score(y_all[te], (p > .5).astype(int)),
                     "n_train": len(tr), "n_test": len(te)})

R = pd.DataFrame(rows)
R.to_csv(os.path.join(FOLDER, "li2026_split_comparison.csv"), index=False)

PAPER = {
    ("CD vs Control", "GSE112057"): 0.90, ("CD vs Control", "GSE119600"): 0.84,
    ("UC vs Control", "GSE112057"): 0.94, ("UC vs Control", "GSE119600"): 0.74,
}
print("=== Setting P (paper split): ours vs paper ===")
sub = R[R["Setting"].str.startswith("P")].copy()
sub["Paper_AUC"] = [PAPER.get((t, d), np.nan) for t, d in zip(sub["Task"], sub["Test_dataset"])]
sub["AUC_95CI"] = ["%.3f [%.3f-%.3f]" % (a, l, h) for a, l, h in zip(sub["AUC"], sub["CI_low"], sub["CI_high"])]
print(sub[["Task", "Test_dataset", "n_train", "n_test", "AUC_95CI", "BalancedAcc", "Paper_AUC"]]
      .round(3).to_string(index=False))
print("\n=== Setting L (our LODO on the 3 datasets, for reference) ===")
sub = R[R["Setting"].str.startswith("L")].copy()
sub["AUC_95CI"] = ["%.3f [%.3f-%.3f]" % (a, l, h) for a, l, h in zip(sub["AUC"], sub["CI_low"], sub["CI_high"])]
print(sub[["Task", "Test_dataset", "n_train", "n_test", "AUC_95CI", "BalancedAcc"]].round(3).to_string(index=False))
