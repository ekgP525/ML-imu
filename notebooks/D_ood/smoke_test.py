"""가짜 임베딩으로 ood.py 파이프라인이 끝까지 도는지 확인한다.

    python notebooks/D_ood/smoke_test.py
"""
import numpy as np

from ood import Prototype, confident_mistakes, run_heldout, summarize, to_results_rows
from splits import make_material_out_folds

rng = np.random.default_rng(42)
MATERIALS = ["carpet", "concrete", "fine_concrete", "hard_tiles",
             "soft_pvc", "soft_tiles", "tiled", "wood"]
N_RECS = [3, 3, 2, 2, 4, 2, 3, 2]  # 합계 21
DIM, PER_REC = 16, 80

centers = rng.normal(0, 3, (len(MATERIALS), DIM))
y, rec, z = [], [], []
r = 0
for m, (name, n) in enumerate(zip(MATERIALS, N_RECS)):
    for _ in range(n):
        offset = rng.normal(0, 0.5, DIM)  # 녹화 간 차이
        z.append(centers[m] + offset + rng.normal(0, 1, (PER_REC, DIM)))
        y += [name] * PER_REC
        rec += [r] * PER_REC
        r += 1
z, y, rec = np.concatenate(z), np.array(y), np.array(rec)
W = rng.normal(0, 1, (DIM, len(MATERIALS) - 1))


def load(heldout, fold):
    return z, z @ W


df = run_heldout(load, y, rec, model="fake", folds=make_material_out_folds(y, rec))
by_material, by_difficulty = summarize(df)
print(by_difficulty.round(3))
print(df[df.method == "proto"][["known_pass", "unknown_reject"]].mean().round(3))

unk = np.where(y == "carpet")[0]
proto = Prototype().fit(z[y != "carpet"], y[y != "carpet"])
print(confident_mistakes(proto, z, y, unk, n=3))
print(to_results_rows(df).head(3).to_string())
assert df.auroc.between(0, 1).all() and df.fpr95.between(0, 1).all()
assert df.groupby("heldout").fold.nunique().tolist() == [21 - n for n in N_RECS]
print("OK", len(df), "rows")
