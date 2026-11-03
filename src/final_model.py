"""Final fraud model: neural network scored by fraud detection rate (FDR).

Usage
-----
    python final_model.py <data_dir> search   # 40 runs with frozen hyperparameters
    python final_model.py <data_dir> <seed>   # final run with one seed (25 in the report)

<data_dir> must contain vars_final.csv: the 20 selected variables in wrapper order,
plus Recnum and Fraud. The data are not public.

Outputs (final run): FDR_trn.csv, FDR_tst.csv, FDR_oot.csv (score-bin results tables)
and Financials_oot.csv (annual fraud $ caught, lost revenue and overall savings).
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

N_VARS = 10                 # top 10 variables in wrapper order
OOT_FIRST_RECNUM = 85519    # first record on Nov 1, 2010
DETECT_RATE = 0.03          # FDR@3%
CLIP = 10                   # cap scaled variables at +/- 10 std dev

# Financial assumptions
GAIN_PER_FRAUD = 500        # $ saved for each fraud caught
LOSS_PER_FP = 40            # $ lost for each good transaction flagged
# OOT is 2 of 12 months; the sample is 100,000 of 10 million transactions/year
ANNUAL_MULT = 12 / 2 * 10_000_000 / 100_000


def load_data(data_dir):
    data = pd.read_csv(f"{data_dir}/vars_final.csv")
    X = data[list(data.columns[:N_VARS])]
    X = (X - X.mean()) / X.std()
    X = X.clip(-CLIP, CLIP)
    X = (X - X.mean()) / X.std()
    y = data["Fraud"]
    split = (data["Recnum"] >= OOT_FIRST_RECNUM).idxmax()
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def make_model(seed):
    # Non-default hyperparameters: hidden_layer_sizes, activation, alpha
    return MLPClassifier(hidden_layer_sizes=(20,), activation="tanh", alpha=0.01, random_state=seed)


def fdr(scores, y, rate=DETECT_RATE):
    """Share of all frauds found in the top `rate` fraction of scores."""
    n_top = int(round(len(scores) * rate))
    top = np.argsort(-scores)[:n_top]
    return y[top].sum() / y.sum()


def fit_one(X_trntst, y_trntst, X_oot, y_oot, seed):
    X_trn, X_tst, y_trn, y_tst = train_test_split(X_trntst, y_trntst, test_size=0.3, random_state=seed)
    model = make_model(seed).fit(X_trn, y_trn)
    sets = {"trn": (X_trn, y_trn), "tst": (X_tst, y_tst), "oot": (X_oot, y_oot)}
    return model, sets


def results_table(scores, y):
    """Population-bin table (1% bins, highest scores first), as in the report."""
    d = pd.DataFrame({"score": scores, "fraud": np.asarray(y)}).sort_values("score", ascending=False)
    n, n_bad = len(d), d["fraud"].sum()
    n_good = n - n_bad
    rows, prev_g, prev_b = [], 0, 0
    for i in range(101):
        k = int(round(n * 0.01 * i))
        cb = int(d["fraud"].iloc[:k].sum())
        cg = k - cb
        r = {"bin": i, "recs": k - prev_g - prev_b, "g": cg - prev_g, "b": cb - prev_b, "tot": k, "cg": cg, "cb": cb}
        if i:
            r["pct_g"] = 100 * r["g"] / r["recs"]
            r["pct_b"] = 100 - r["pct_g"]
            r["pct_cg"] = 100 * cg / n_good
            r["FDR"] = 100 * cb / n_bad
            r["KS"] = r["FDR"] - r["pct_cg"]
            r["FPR"] = cg / cb if cb else np.nan
        rows.append(r)
        prev_g, prev_b = cg, cb
    return pd.DataFrame(rows)


def main():
    data_dir, mode = sys.argv[1], sys.argv[2]
    X_trntst, y_trntst, X_oot, y_oot = load_data(data_dir)

    if mode == "search":
        rows = []
        for seed in range(40):
            model, sets = fit_one(X_trntst, y_trntst, X_oot, y_oot, seed)
            r = {"seed": seed}
            for name, (X, y) in sets.items():
                r[name] = fdr(model.predict_proba(X)[:, 1], y.values)
            rows.append(r)
            print(r, flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(f"{data_dir}/search.csv", index=False)
        print(df[["trn", "tst", "oot"]].describe().round(3))
        return

    seed = int(mode)
    model, sets = fit_one(X_trntst, y_trntst, X_oot, y_oot, seed)
    summary = {}
    for name, (X, y) in sets.items():
        table = results_table(model.predict_proba(X)[:, 1], y)
        table.to_csv(f"{data_dir}/FDR_{name}.csv", index=False)
        summary[name] = round(table.loc[3, "FDR"], 2)
    print("FDR@3% (%):", json.dumps(summary))

    oot = pd.read_csv(f"{data_dir}/FDR_oot.csv")
    fin = pd.DataFrame({
        "bin": oot["bin"],
        "Fraud Savings": oot["cb"] * GAIN_PER_FRAUD * ANNUAL_MULT,
        "FP Loss": oot["cg"] * LOSS_PER_FP * ANNUAL_MULT,
    })
    fin["Overall Savings"] = fin["Fraud Savings"] - fin["FP Loss"]
    fin.to_csv(f"{data_dir}/Financials_oot.csv", index=False)
    best = fin.loc[fin["Overall Savings"].idxmax()]
    print(f"Max annual savings ${best['Overall Savings']:,.0f} at {int(best['bin'])}% cutoff")


if __name__ == "__main__":
    main()
