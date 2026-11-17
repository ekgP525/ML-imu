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

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.evaluate import auroc, fpr_at_95_tpr  # noqa: E402

SEED = 42
TPR = 0.95

# 쉬운(far)/어려운(near) 미등록 재질. 정리 문서 5장 기준, 나머지는 mid
DIFFICULTY = {
    "carpet": "far",
    "concrete": "near",
    "tiled": "near",
    "hard_tiles": "near",
}

RESULT_COLUMNS = [
    "date", "owner", "model", "setting", "fold", "macro_f1", "accuracy",
    "mode_macro_f1", "auroc", "fpr95", "note",
]


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


def cross_fit_scores(make_scorer, z, y, rec):
    """학습 녹화를 하나씩 검증 녹화로 떼어 그 녹화 구간의 점수를 모은다.

    떼어내면 재질이 학습에서 사라지는 녹화(재질의 마지막 학습 녹화)는 건너뛴다.
    """
    scores = []
    for r in np.unique(rec):
        val = rec == r
        if not np.any(y[~val] == y[val][0]):
            continue
        scorer = make_scorer().fit(z[~val], y[~val])
        scores.append(scorer.score(z[val]))
    return np.concatenate(scores)


def open_set_predict(proto, z, tau, unknown=-1):
    """τ를 넘으면 미등록(unknown), 아니면 가장 가까운 재질."""
    pred = proto.predict(z).astype(object)
    pred[proto.score(z) > tau] = unknown
    return pred


# ---------------------------------------------------------------- 실험 루프
def heldout_folds(y, rec):
    """재질 하나 빼기 × 남은 녹화 하나 빼기.

    fold 번호는 테스트 녹화 번호(splits/folds.json과 같은 번호)를 그대로 쓴다.
    """
    rec_material = {r: y[rec == r][0] for r in np.unique(rec)}
    for m in np.unique(y):
        remaining = [r for r, c in rec_material.items() if c != m]
        for r in remaining:
            yield {
                "heldout": m,
                "fold": int(r),
                "test_rec": r,
                "train_recs": [t for t in remaining if t != r],
            }


def run_fold(z, y, rec, split, logits=None, knn_k=10):
    """한 fold에서 모든 방법의 AUROC·FPR@95와 제안 방법의 τ 판정 결과를 계산한다."""
    train = np.isin(rec, split["train_recs"])
    known = rec == split["test_rec"]
    unknown = y == split["heldout"]
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
    tau = fit_tau(cross_fit_scores(Prototype, z[train], y[train], rec[train]))
    s = scores["proto"]
    next(r for r in rows if r["method"] == "proto").update({
        "tau": tau,
        "known_pass": float(np.mean(s[y_ood == 0] <= tau)),
        "unknown_reject": float(np.mean(s[y_ood == 1] > tau)),
    })
    return rows


def run_heldout(load, y, rec, model, knn_k=10):
    """load(heldout, fold) -> (z, logits 또는 None). fold별 결과 표를 반환한다."""
    rows = []
    for split in heldout_folds(y, rec):
        z, logits = load(split["heldout"], split["fold"])
        for row in run_fold(z, y, rec, split, logits, knn_k):
            rows.append({
                "model": model,
                "heldout": split["heldout"],
                "difficulty": DIFFICULTY.get(split["heldout"], "mid"),
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
