
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
def combat_parametric(Y, batch, max_iter=500, tol=1e-4):

    Y = np.asarray(Y, dtype=float)
    batch = np.asarray(batch)
    G, N = Y.shape
    levels = list(dict.fromkeys(batch))
    idx = [np.where(batch == b)[0] for b in levels]
    n_b = np.array([len(i) for i in idx], dtype=float)
    batch_mean = np.stack([Y[:, i].mean(axis=1) for i in idx], axis=1)
    grand = (batch_mean * (n_b / N)).sum(axis=1)[:, None]
    resid = np.concatenate([Y[:, i] - batch_mean[:, [k]] for k, i in enumerate(idx)], axis=1)
    var_pooled = (resid ** 2).sum(axis=1)[:, None] / N
    S = (Y - grand) / np.sqrt(var_pooled)
    gamma_hat = np.stack([S[:, i].mean(axis=1) for i in idx], axis=1)
    delta_hat = np.stack([S[:, i].var(axis=1, ddof=1) for i in idx], axis=1)
    gamma_bar = gamma_hat.mean(axis=0)
    t2 = gamma_hat.var(axis=0, ddof=1)
    m = delta_hat.mean(axis=0)
    s2 = delta_hat.var(axis=0, ddof=1)
    a_prior = (2 * s2 + m ** 2) / s2
    b_prior = (m * s2 + m ** 3) / s2
    adj = np.empty_like(Y)
    for k, i in enumerate(idx):
        n = n_b[k]
        g_old, d_old = gamma_hat[:, k].copy(), delta_hat[:, k].copy()
        for _ in range(max_iter):
            g_new = (t2[k] * n * gamma_hat[:, k] + d_old * gamma_bar[k]) / (t2[k] * n + d_old)
            sum2 = ((S[:, i] - g_new[:, None]) ** 2).sum(axis=1)
            d_new = (0.5 * sum2 + b_prior[k]) / (n / 2.0 + a_prior[k] - 1.0)
            change = max(np.max(np.abs(g_new - g_old) / np.abs(g_old).clip(1e-12)),
                         np.max(np.abs(d_new - d_old) / np.abs(d_old).clip(1e-12)))
            g_old, d_old = g_new, d_new
            if change < tol:
                break

        adj[:, i] = (S[:, i] - g_old[:, None]) / np.sqrt(d_old)[:, None]
    return adj * np.sqrt(var_pooled) + grand


FOLDER = r"CBIC"

SEED, N_BOOT, K, C_VALUE = 42, 1000, 500, 0.01

Z = pd.read_csv(os.path.join(FOLDER, "combined5_zscore_expression.csv.gz"), index_col=0)
L = pd.read_csv(os.path.join(FOLDER, "combined5_log2_expression.csv.gz"), index_col=0)
meta = pd.read_csv(os.path.join(FOLDER, "combined5_metadata.csv"))
assert list(Z.columns) == list(L.columns) == list(meta["Sample"])
grp, dsn = meta["Group"].values, meta["Dataset"].values
datasets = list(dict.fromkeys(dsn))
print("Genes:", L.shape[0], "| samples:", L.shape[1])

ok = np.ones(L.shape[0], bool)
for d in datasets:
    ok &= (L.loc[:, dsn == d].std(axis=1).values > 1e-8)
print("Genes kept for ComBat (variable in every dataset): %d of %d" % (ok.sum(), len(ok)))
Lc = L.loc[ok]
print("Running ComBat (parametric empirical Bayes, batch = dataset, no covariates) ...")
C = pd.DataFrame(combat_parametric(Lc.values, dsn), index=Lc.index, columns=Lc.columns)

MATRICES = {"log2 only": L.loc[Lc.index], "per-dataset z-score (HBC-IBD)": Z.loc[Lc.index], "ComBat": C}
rng = np.random.RandomState(SEED)


def boot_ci(yt, p):
    a = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(yt), len(yt))
        if len(set(yt[i])) == 2:
            a.append(roc_auc_score(yt[i], p[i]))
    return np.percentile(a, [2.5, 97.5])


rows = []
for level, use, pos in (("Level 1: IBD vs Control", np.ones(len(grp), bool), lambda g: g != "Control"),
                        ("Level 2: CD vs UC", grp != "Control", lambda g: g == "CD")):
    y_all = pos(grp).astype(int)
    for mname, M in MATRICES.items():
        X_all = M.T.values
        for test in datasets:
            tr = use & (dsn != test)
            te = use & (dsn == test)
            if len(set(y_all[te])) < 2:
                continue
            sel = SelectKBest(f_classif, k=K).fit(X_all[tr], y_all[tr])
            sc = StandardScaler().fit(sel.transform(X_all[tr]))
            clf = LogisticRegression(C=C_VALUE, class_weight="balanced", max_iter=5000, random_state=SEED)
            clf.fit(sc.transform(sel.transform(X_all[tr])), y_all[tr])
            p = clf.predict_proba(sc.transform(sel.transform(X_all[te])))[:, 1]
            lo, hi = boot_ci(y_all[te], p)
            rows.append({"Level": level, "Method": mname, "Test_dataset": test, "AUC": roc_auc_score(y_all[te], p),
                         "CI_low": lo, "CI_high": hi,
                         "BalancedAcc": balanced_accuracy_score(y_all[te], (p > 0.5).astype(int)),
                         "n_test": int(te.sum())})
R = pd.DataFrame(rows)
R.to_csv(os.path.join(FOLDER, "combat_comparison_results.csv"), index=False)
pd.set_option("display.width", 200)
print("\n=== AUC per held-out cohort ===")
print(R.pivot_table(index=["Level", "Test_dataset"], columns="Method", values="AUC").round(3).to_string())
print("\n=== Mean AUC and mean balanced accuracy over held-out cohorts ===")
print(R.groupby(["Level", "Method"]).agg(mean_AUC=("AUC", "mean"), mean_BalAcc=("BalancedAcc", "mean"),
                                         n_cohorts=("AUC", "size")).round(3).to_string())
