"""D. 미등록 바닥 판별: OOD 점수, τ 설정, 재질 하나 빼기 실험.

모든 점수는 "클수록 미등록"으로 통일한다.
임베딩·logits는 데이터셋 구간 순서(7626개)와 같은 순서라고 가정한다.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from evaluate import RESULT_COLUMNS, auroc, fpr_at_95_tpr  # noqa: E402
from splits import inner_val_recordings, load_material_folds  # noqa: E402

SEED = 42
TPR = 0.95

# 쉬운(far)/어려운(near) 미등록 재질. 정리 문서 5장 기준, 나머지는 mid
DIFFICULTY = {
    "carpet": "far",
    "concrete": "near",
    "tiled": "near",
    "hard_tiles": "near",
}


# ---------------------------------------------------------------- 점수 함수
class Prototype:
    """제안 방법: 가장 가까운 재질 프로토타입(평균 임베딩)까지의 거리."""

    def fit(self, z, y):
        self.classes_ = np.unique(y)
        self.mu_ = np.stack([z[y == c].mean(axis=0) for c in self.classes_])
        return self

    def _dist(self, z):
        return np.linalg.norm(z[:, None, :] - self.mu_[None], axis=2)

    def score(self, z):
        return self._dist(z).min(axis=1)

    def predict(self, z):
        return self.classes_[self._dist(z).argmin(axis=1)]


class Mahalanobis:
    """Lee et al., 2018: 재질별 평균 + 공유 공분산의 최소 마할라노비스 거리."""

    def __init__(self, reg=1e-6):
        self.reg = reg

    def fit(self, z, y):
        self.classes_ = np.unique(y)
        self.mu_ = np.stack([z[y == c].mean(axis=0) for c in self.classes_])
        centered = z - self.mu_[np.searchsorted(self.classes_, y)]
        cov = centered.T @ centered / len(z)
        self.prec_ = np.linalg.inv(cov + self.reg * np.eye(cov.shape[0]))
        return self

    def score(self, z):
        diff = z[:, None, :] - self.mu_[None]
        d2 = np.einsum("ncd,de,nce->nc", diff, self.prec_, diff)
        return d2.min(axis=1)


class KNN:
    """Sun et al., 2022: L2 정규화한 임베딩에서 k번째 이웃까지의 거리."""

    def __init__(self, k=10):
        self.k = k

    @staticmethod
    def _norm(z):
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

    def fit(self, z, y=None):
        self.bank_ = self._norm(z)
        return self

    def score(self, z):
        d = np.sqrt(np.maximum(2 - 2 * self._norm(z) @ self.bank_.T, 0))
        return np.partition(d, self.k - 1, axis=1)[:, self.k - 1]


def msp_score(logits):
    """Hendrycks & Gimpel, 2017: softmax 최대 확률 (음수로 바꿔 클수록 미등록)."""
    return -softmax(logits, axis=1).max(axis=1)


def energy_score(logits, T=1.0):
    """Liu et al., 2020: E(x) = -T·logsumexp(f(x)/T)."""
    return -T * logsumexp(logits / T, axis=1)


EMBEDDING_SCORERS = {
    "proto": Prototype,
    "maha": Mahalanobis,
    "knn": KNN,
}
LOGIT_SCORERS = {
    "msp": msp_score,
    "energy": energy_score,
}


# ---------------------------------------------------------------- τ와 판정
def fit_tau(known_scores, tpr=TPR):
    """알려진 재질 검증 구간의 tpr 비율이 통과하는 거리."""
    return float(np.quantile(known_scores, tpr))


def val_scores(make_scorer, z, y, rec, train_recs):
    """학습 녹화 중 검증 녹화(splits.inner_val_recordings)를 떼고 맞춘 뒤 검증 구간 점수를 낸다."""
    val_recs = inner_val_recordings(train_recs, y, rec)
    fit = np.isin(rec, train_recs) & ~np.isin(rec, val_recs)
    val = np.isin(rec, val_recs)
    return make_scorer().fit(z[fit], y[fit]).score(z[val])


def open_set_predict(proto, z, tau, unknown=-1):
    """τ를 넘으면 미등록(unknown), 아니면 가장 가까운 재질."""
    pred = proto.predict(z).astype(object)
    pred[proto.score(z) > tau] = unknown
    return pred


# ---------------------------------------------------------------- 실험 루프
def run_fold(z, y, rec, split, logits=None, knn_k=10):
    """한 fold에서 모든 방법의 AUROC·FPR@95와 제안 방법의 τ 판정 결과를 계산한다."""
    train = np.isin(rec, split["train"])
    known = np.isin(rec, split["test_known"])
    unknown = np.isin(rec, split["unknown"])
    test = known | unknown
    y_ood = unknown[test].astype(int)

    scores = {}
    for name, cls in EMBEDDING_SCORERS.items():
        make = (lambda: cls(k=knn_k)) if name == "knn" else cls
        scores[name] = make().fit(z[train], y[train]).score(z[test])
    if logits is not None:
        for name, fn in LOGIT_SCORERS.items():
            scores[name] = fn(logits[test])

    rows = [
        {"method": name, "auroc": auroc(y_ood, s), "fpr95": fpr_at_95_tpr(y_ood, s)}
        for name, s in scores.items()
    ]

    # 제안 방법: τ는 학습 녹화 안의 검증 녹화로만 정한다
    tau = fit_tau(val_scores(Prototype, z, y, rec, split["train"]))
    s = scores["proto"]
    next(r for r in rows if r["method"] == "proto").update({
        "tau": tau,
        "known_pass": float(np.mean(s[y_ood == 0] <= tau)),
        "unknown_reject": float(np.mean(s[y_ood == 1] > tau)),
    })
    return rows


def run_heldout(load, y, rec, model, folds=None, knn_k=10):
    """load(held_out, fold) -> (z, logits 또는 None). fold별 결과 표를 반환한다.

    folds를 주지 않으면 splits/folds.json의 material_out(147개)을 쓴다.
    """
    folds = load_material_folds() if folds is None else folds
    rows = []
    for split in folds:
        z, logits = load(split["held_out"], split["fold"])
        for row in run_fold(z, y, rec, split, logits, knn_k):
            rows.append({
                "model": model,
                "heldout": split["held_out"],
                "difficulty": DIFFICULTY.get(split["held_out"], "mid"),
                "fold": split["fold"],
                **row,
            })
    return pd.DataFrame(rows)


def summarize(df):
    """방법 × 빠진 재질별 평균·표준편차, 방법 × 난이도별 평균."""
    by_material = df.groupby(["method", "heldout"])[["auroc", "fpr95"]].agg(["mean", "std"])
    by_difficulty = df.groupby(["method", "difficulty"])[["auroc", "fpr95"]].mean()
    return by_material, by_difficulty


def confident_mistakes(proto, z, y, idx_unknown, n=20):
    """미등록 구간 중 가장 확신 있게 기존 재질로 받아들인 사례 (점수 낮은 순)."""
    s = proto.score(z[idx_unknown])
    order = np.argsort(s)[:n]
    return pd.DataFrame({
        "index": idx_unknown[order],
        "true": y[idx_unknown[order]],
        "pred": proto.predict(z[idx_unknown[order]]),
        "score": s[order],
    })


def to_results_rows(df, owner="D", note=""):
    """results/results.csv 형식으로 변환한다."""
    out = pd.DataFrame({
        "date": date.today().isoformat(),
        "owner": owner,
        "model": df["model"] + "+" + df["method"],
        "setting": "heldout=" + df["heldout"].astype(str),
        "fold": df["fold"],
        "auroc": df["auroc"],
        "fpr95": df["fpr95"],
        "note": note,
    })
    return out.reindex(columns=RESULT_COLUMNS)
