
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

FOLDER = r"CBIC"


NAMES = ["GSE112057", "GSE119600", "GSE186507"]
SETS = {"Interferon Gamma Response": "IFN_gamma", "Interferon Alpha Response": "IFN_alpha"}
ACTIVITY_COL = "ibd_clinicianmeasure_inactive_active"
SEVERITY_COL = "ibd_endoseverity_4levels"

gene_sets = {}
try:
    import gseapy
    lib = gseapy.get_library("MSigDB_Hallmark_2020")
    for key, short in SETS.items():
        gene_sets[short] = [g.upper() for g in lib[key]]
    print("Hallmark sets loaded through gseapy.")
except Exception as e:
    local = os.path.join(FOLDER, "hallmark_ifn_genes.csv")
    if os.path.exists(local):
        df = pd.read_csv(local)
        for s, sub in df.groupby("Set"):
            gene_sets[s] = [g.upper() for g in sub["Gene"]]
        print("Gene sets read from", local)
    else:
        raise SystemExit("Could not load Hallmark sets (%s). Install gseapy (pip install gseapy) and check the "
                         "internet, or provide hallmark_ifn_genes.csv (Set,Gene)." % type(e).__name__)

z = pd.read_csv(os.path.join(FOLDER, "combined_zscore_expression.csv.gz"), index_col=0)
z.index = z.index.str.upper()
meta = pd.read_csv(os.path.join(FOLDER, "combined_metadata.csv"))
assert list(z.columns) == list(meta["Sample"])
scores = pd.read_csv(os.path.join(FOLDER, "sample_scores.csv")).set_index("Sample")

module_score = {}
for name, genes in gene_sets.items():
    present = [g for g in dict.fromkeys(genes) if g in z.index]
    print("%s: %d of %d genes present in our data" % (name, len(present), len(set(genes))))
    module_score[name] = z.loc[present].mean(axis=0)


def covariates(df, groups):
    cols = [c for c in df.columns if re.search(r"(^|_)age($|_)", c.lower()) or c.lower().startswith("age")
            or c.lower() in ("gender", "sex")]
    parts, used = [], []
    for c in cols:
        raw = df[c].replace("", np.nan)
        if max(float(raw[groups == g].isna().mean()) for g in np.unique(groups)) > 0.05:
            continue
        num = pd.to_numeric(raw, errors="coerce")
        if num.notna().sum() > 0 and num.notna().sum() >= 0.9 * raw.notna().sum():
            v = num.fillna(num.median())
            if v.std() > 0:
                parts.append(pd.DataFrame({c: (v - v.mean()) / v.std()}, index=df.index))
                used.append(c)
        else:
            d = pd.get_dummies(raw.fillna(raw.mode().iloc[0]).astype(str), prefix=c, drop_first=True).astype(float)
            d = d.loc[:, d.std() > 0]
            parts.append(d)
            used += list(d.columns)
    return (pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)), used


def ols_cd(y, cd, X):
    D = np.column_stack([np.ones(len(y)), cd.astype(float), X]) if X.shape[1] else np.column_stack([np.ones(len(y)), cd.astype(float)])
    XtX_inv = np.linalg.pinv(D.T @ D)
    beta = XtX_inv @ D.T @ y
    resid = y - D @ beta
    dof = len(y) - np.linalg.matrix_rank(D)
    se = np.sqrt((resid ** 2).sum() / dof * XtX_inv[1, 1])
    tcrit = stats.t.isf(0.025, dof)
    return beta[1], se, 2 * stats.t.sf(abs(beta[1] / se), dof), beta[1] - tcrit * se, beta[1] + tcrit * se


rows = []
adjusted_for_meta = {k: [] for k in module_score}
for ds in NAMES:
    sm = meta[meta["Dataset"] == ds].reset_index(drop=True)
    pat = (sm["Group"] != "Control").values
    sm = sm[pat].reset_index(drop=True)
    md = pd.read_csv(os.path.join(FOLDER, "%s_metadata_CD_UC_Control.csv" % ds), dtype=str)
    md["Sample"] = ds + "|" + md["GSM"].str.strip()
    md = md.drop_duplicates("Sample").set_index("Sample").loc[sm["Sample"]]
    cd = (sm["Group"] == "CD").values.astype(int)
    C, used = covariates(md, sm["Group"].values)
    C = C.reset_index(drop=True)
    S = scores.loc[sm["Sample"], ["Neutrophil", "Lymphocyte"]].reset_index(drop=True)
    S = (S - S.mean()) / S.std()
    base_cov = pd.concat([C, S], axis=1)
    print("\n=== %s: CD=%d, UC=%d | covariates: %s + Neutrophil + Lymphocyte ===" % (ds, cd.sum(), (1 - cd).sum(), used or "none"))
    for name in module_score:
        y = module_score[name].loc[sm["Sample"]].values
        y = (y - y.mean()) / y.std()
        d_ = (y[cd == 1].mean() - y[cd == 0].mean()) / np.sqrt(((cd.sum() - 1) * y[cd == 1].var(ddof=1) + ((1 - cd).sum() - 1) * y[cd == 0].var(ddof=1)) / (len(y) - 2))
        auc = roc_auc_score(cd, y)
        p_un = stats.ttest_ind(y[cd == 1], y[cd == 0], equal_var=False).pvalue
        b, se, p, lo, hi = ols_cd(y, cd, base_cov.values)
        print("  %-10s unadjusted: d=%+.2f AUC=%.2f p=%.1e | adjusted: %+.2f SD [%+.2f, %+.2f] p=%.1e"
              % (name, d_, auc, p_un, b, lo, hi, p))
        rows.append({"Dataset": ds, "Module": name, "Model": "adjusted_age_sex_neutro_lympho", "n": len(y),
                     "d_unadjusted": round(d_, 3), "AUC_unadjusted": round(auc, 3), "Coef_SD": round(b, 3),
                     "SE": round(se, 3), "CI_low": round(lo, 3), "CI_high": round(hi, 3), "p": p})
        adjusted_for_meta[name].append((b, se))
        if ds == "GSE186507":
            act = md[ACTIVITY_COL].replace("", np.nan).reset_index(drop=True)
            sev = md[SEVERITY_COL].replace("", np.nan).reset_index(drop=True)
            ok = act.notna().values & sev.notna().values
            extra = pd.concat([pd.get_dummies(act[ok], prefix="act", drop_first=True),
                               pd.get_dummies(sev[ok], prefix="sev", drop_first=True)], axis=1).astype(float)
            X2 = np.column_stack([base_cov.values[ok], extra.values])
            b2, se2, p2, lo2, hi2 = ols_cd(y[ok], cd[ok], X2)
            b1, se1, p1, lo1, hi1 = ols_cd(y[ok], cd[ok], base_cov.values[ok])
            print("  %-10s complete cases n=%d (CD=%d, UC=%d): before activity/severity %+.2f SD [%+.2f, %+.2f] p=%.1e"
                  % (name, ok.sum(), cd[ok].sum(), (1 - cd[ok]).sum(), b1, lo1, hi1, p1))
            print("  %-10s                              + activity + severity: %+.2f SD [%+.2f, %+.2f] p=%.1e   <-- key line"
                  % ("", b2, lo2, hi2, p2))
            rows.append({"Dataset": ds, "Module": name, "Model": "+activity+severity (complete cases)", "n": int(ok.sum()),
                         "Coef_SD": round(b2, 3), "SE": round(se2, 3), "CI_low": round(lo2, 3), "CI_high": round(hi2, 3), "p": p2})

for name, est in adjusted_for_meta.items():
    b = np.array([e[0] for e in est])
    se = np.array([e[1] for e in est])
    w = 1 / se ** 2
    bm = (w * b).sum() / w.sum()
    sem = np.sqrt(1 / w.sum())
    q = (w * (b - bm) ** 2).sum()
    print("  %-10s combined %+.2f SD [%+.2f, %+.2f], p=%.1e | heterogeneity Q p=%.2f (small p = datasets disagree)"
          % (name, bm, bm - 1.96 * sem, bm + 1.96 * sem, 2 * stats.norm.sf(abs(bm / sem)), stats.chi2.sf(q, len(b) - 1)))
    rows.append({"Dataset": "META", "Module": name, "Model": "fixed_effect", "n": "", "Coef_SD": round(bm, 3),
                 "SE": round(sem, 3), "CI_low": round(bm - 1.96 * sem, 3), "CI_high": round(bm + 1.96 * sem, 3),
                 "p": 2 * stats.norm.sf(abs(bm / sem))})
pd.DataFrame(rows).to_csv(os.path.join(FOLDER, "ifn_module_results.csv"), index=False)