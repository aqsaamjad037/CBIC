

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import gseapy as gp

warnings.filterwarnings("ignore")

FOLDER = r"CBIC"
K, C_VALUE, SEED = 500, 0.01, 42
OUT = os.path.join(FOLDER, "biology_results_5cohort")
os.makedirs(OUT, exist_ok=True)

Z = pd.read_csv(os.path.join(FOLDER, "combined5_zscore_expression.csv.gz"), index_col=0)
meta = pd.read_csv(os.path.join(FOLDER, "combined5_metadata.csv"))
assert list(Z.columns) == list(meta["Sample"]), "Sample order differs between expression and metadata"
genes = np.array(Z.index.astype(str))
X_all = Z.T.values
grp = meta["Group"].values
dsn = meta["Dataset"].values
datasets = sorted(set(dsn))

TASKS = {
    "Level1_IBD_vs_Control": dict(use=np.ones(len(grp), bool), pos=lambda g: g != "Control",
                                  pos_name="IBD", neg_name="Control"),
    "Level2_CD_vs_UC": dict(use=(grp != "Control"), pos=lambda g: g == "CD",
                            pos_name="CD", neg_name="UC"),
}

stable_tables = {}
for task, cfg in TASKS.items():
    use, y_all = cfg["use"], cfg["pos"](grp).astype(int)
    sel_count = pd.Series(0, index=genes, dtype=int)
    coef_sum = pd.Series(0.0, index=genes)
    n_folds = 0
    for held in datasets:
        tr = use & (dsn != held)
        te = use & (dsn == held)
        if len(set(y_all[te])) < 2 or len(set(y_all[tr])) < 2:
            continue
        sel = SelectKBest(f_classif, k=K).fit(X_all[tr], y_all[tr])
        idx = sel.get_support(indices=True)
        sc = StandardScaler().fit(X_all[tr][:, idx])
        clf = LogisticRegression(penalty="l2", C=C_VALUE, class_weight="balanced",
                                 max_iter=5000, random_state=SEED)
        clf.fit(sc.transform(X_all[tr][:, idx]), y_all[tr])
        sel_count.iloc[idx] += 1
        coef_sum.iloc[idx] += clf.coef_[0]
        n_folds += 1
    tab = pd.DataFrame({"gene": genes, "folds_selected": sel_count.values,
                        "mean_coef": (coef_sum / sel_count.replace(0, np.nan)).values})
    tab["direction"] = np.where(tab["mean_coef"] > 0, "higher in " + cfg["pos_name"],
                                "higher in " + cfg["neg_name"])
    tab = tab[tab["folds_selected"] > 0].sort_values(["folds_selected", "mean_coef"], ascending=[False, False])
    tab.to_csv(os.path.join(OUT, "%s_gene_frequency.csv" % task), index=False)
    stable = tab[tab["folds_selected"] == n_folds].copy()
    stable.to_csv(os.path.join(OUT, "%s_stable_genes.csv" % task), index=False)
    stable_tables[task] = stable
    print("%s: %d valid folds, %d genes selected in every fold (up in %s: %d, up in %s: %d)" % (
        task, n_folds, len(stable), cfg["pos_name"], (stable["mean_coef"] > 0).sum(),
        cfg["neg_name"], (stable["mean_coef"] < 0).sum()))

LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"]
results = {}
for task, stable in stable_tables.items():
    if len(stable) < 10:
        print("%s: fewer than 10 stable genes, enrichment skipped." % task)
        continue
    try:
        try:
            enr = gp.enrichr(gene_list=list(stable["gene"]), gene_sets=LIBRARIES,
                             background=list(genes), organism="human", outdir=None)
            res = enr.results
            if "Overlap" not in res.columns or res.empty:
                raise ValueError("background run returned no usable table")
            bg_note = "background = all genes available to the model"
        except Exception:
            enr = gp.enrichr(gene_list=list(stable["gene"]), gene_sets=LIBRARIES,
                             organism="human", outdir=None)
            res = enr.results
            bg_note = "default Enrichr background (whole genome)"
        print("  (%s)" % bg_note)
        keep = [c for c in ["Gene_set", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"] if c in res.columns]
        res = res[keep].sort_values("Adjusted P-value")
        res.to_csv(os.path.join(OUT, "%s_enrichment_all.csv" % task), index=False)
        results[task] = res
        print("\n%s - top 10 terms (adjusted P-value = Benjamini-Hochberg FDR):" % task)
        print(res.head(10)[[c for c in ["Gene_set", "Term", "Overlap", "Adjusted P-value"] if c in res.columns]].to_string(index=False))
    except Exception as e:
        print("Enrichment failed for %s: %s" % (task, e))

if results:
    fig, axes = plt.subplots(1, len(results), figsize=(7.5 * len(results), 5.6), squeeze=False)
    for ax, (task, res) in zip(axes[0], results.items()):
        top = res.head(10).iloc[::-1]
        vals = -np.log10(top["Adjusted P-value"].clip(lower=1e-300))
        labels = [t.split(" (GO:")[0][:55] for t in top["Term"]]
        ax.barh(range(len(top)), vals, color="#2a78d6" if task.startswith("Level1") else "#eb6834")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(-np.log10(0.05), color="#52514e", ls="--", lw=1)
        ax.set_xlabel("-log10 adjusted P-value")
        ax.set_title(task.replace("_", " "), fontsize=11, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "Figure_biology_enrichment.png"), dpi=300, bbox_inches="tight")
    print("\nFigure saved in", OUT)
