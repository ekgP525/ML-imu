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


class ScaledPrototype(Prototype):
    """프로토타입 거리를 재질별 학습 구간 거리 중앙값으로 나눈 값의 최솟값.

    조용한 재질은 원래 좁게 뭉쳐 있어서 같은 거리도 더 크게 친다 (진동 세기와 분리).
    """

    def fit(self, z, y):
        super().fit(z, y)
        d = self._dist(z)
        self.scale_ = np.array([np.median(d[y == c, i]) for i, c in enumerate(self.classes_)])
        return self

    def _dist(self, z):
        d = super()._dist(z)
        return d / self.scale_ if hasattr(self, "scale_") else d


class NearestScaledPrototype(ScaledPrototype):
    """원래 거리로 가장 가까운 재질을 먼저 고른 뒤, 그 재질의 퍼짐으로만 나눈 거리.

    ScaledPrototype은 넓게 퍼진(덜컹거리는) 재질이 나누는 값이 커서 조용한 미등록 구간을 흡수한다.
    """

    def _nearest(self, z):
        return Prototype._dist(self, z).argmin(axis=1)

    def score(self, z):
        d = Prototype._dist(self, z)
        i = d.argmin(axis=1)
        return d[np.arange(len(z)), i] / self.scale_[i]

    def predict(self, z):
        return self.classes_[self._nearest(z)]


class ClassMahalanobis:
    """재질마다 따로 구한 공분산(Ledoit-Wolf 수축)의 최소 마할라노비스 거리."""

    def fit(self, z, y):
        from sklearn.covariance import LedoitWolf
        self.classes_ = np.unique(y)
        self.mu_, self.prec_ = [], []
        for c in self.classes_:
            lw = LedoitWolf().fit(z[y == c])
            self.mu_.append(lw.location_)
            self.prec_.append(lw.precision_)
        return self

    def score(self, z):
        d2 = np.stack([np.einsum("nd,de,ne->n", z - mu, prec, z - mu)
                       for mu, prec in zip(self.mu_, self.prec_)], axis=1)
        return d2.min(axis=1)


class RatioPrototype(Prototype):
    """가장 가까운 / 두 번째로 가까운 프로토타입 거리 비율. 1에 가까울수록 어느 재질인지 애매하다."""

    def score(self, z):
        d = np.sort(self._dist(z), axis=1)
        return d[:, 0] / (d[:, 1] + 1e-12)


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
# 진동 세기와 거리를 떼어 놓으려는 점수 (D-03)
SCALE_FREE_SCORERS = {
    "scaled": ScaledPrototype,
    "scaled_nn": NearestScaledPrototype,
    "cmaha": ClassMahalanobis,
    "ratio": RatioPrototype,
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
def run_fold(z, y, rec, split, logits=None, knn_k=10, scorers=None, tau_methods=("proto",), vib=None):
    """한 fold에서 모든 방법의 AUROC·FPR@95와 tau_methods의 τ 판정 결과를 계산한다.

    vib(구간별 진동 세기)를 주면 테스트 구간에서 점수와 진동의 Spearman 상관(rho_vib)도 계산한다.
    """
    from scipy.stats import spearmanr
    scorers = EMBEDDING_SCORERS if scorers is None else scorers
    train = np.isin(rec, split["train"])
    known = np.isin(rec, split["test_known"])
    unknown = np.isin(rec, split["unknown"])
    test = known | unknown
    y_ood = unknown[test].astype(int)

    scores = {}
    for name, cls in scorers.items():
        make = (lambda: cls(k=knn_k)) if name == "knn" else cls
        scores[name] = make().fit(z[train], y[train]).score(z[test])
    if logits is not None:
        for name, fn in LOGIT_SCORERS.items():
            scores[name] = fn(logits[test])

    rows = [
        {"method": name, "auroc": auroc(y_ood, s), "fpr95": fpr_at_95_tpr(y_ood, s)}
        for name, s in scores.items()
    ]

    if vib is not None:
        for r in rows:
            r["rho_vib"] = float(spearmanr(scores[r["method"]], vib[test])[0])

    # τ는 학습 녹화 안의 검증 녹화로만 정한다
    for name in tau_methods:
        tau = fit_tau(val_scores(scorers[name], z, y, rec, split["train"]))
        s = scores[name]
        next(r for r in rows if r["method"] == name).update({
            "tau": tau,
            "known_pass": float(np.mean(s[y_ood == 0] <= tau)),
            "unknown_reject": float(np.mean(s[y_ood == 1] > tau)),
        })
    return rows


def run_heldout(load, y, rec, model, folds=None, knn_k=10, **fold_kw):
    """load(held_out, fold) -> (z, logits 또는 None). fold별 결과 표를 반환한다.

    folds를 주지 않으면 splits/folds.json의 material_out(147개)을 쓴다.
    """
    folds = load_material_folds() if folds is None else folds
    rows = []
    for split in folds:
        z, logits = load(split["held_out"], split["fold"])
        for row in run_fold(z, y, rec, split, logits, knn_k, **fold_kw):
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


# ---------------------------------------------------------------- 원인 진단
def diagnose_fold(z, y, rec, split):
    """프로토타입 거리를 학습 / 알려진 테스트 / 빠진 재질 세 그룹으로 나눠 비교한다.

    학습 vs 빠진 재질 AUROC가 높고 학습 vs 알려진 테스트 AUROC도 높으면,
    임베딩은 미등록을 구분하지만 처음 보는 녹화도 똑같이 낯설게 본다(녹화 차이)는 뜻이다.
    """
    train = np.isin(rec, split["train"])
    known = np.isin(rec, split["test_known"])
    unknown = np.isin(rec, split["unknown"])
    proto = Prototype().fit(z[train], y[train])
    d = {g: proto.score(z[m]) for g, m in [("train", train), ("known", known), ("unknown", unknown)]}

    def pair_auroc(a, b):
        return auroc(np.r_[np.zeros(len(d[a])), np.ones(len(d[b]))], np.r_[d[a], d[b]])

    return {
        "med_train": float(np.median(d["train"])),
        "med_known": float(np.median(d["known"])),
        "med_unknown": float(np.median(d["unknown"])),
        "auroc_train_unknown": pair_auroc("train", "unknown"),
        "auroc_train_known": pair_auroc("train", "known"),
        "auroc_known_unknown": pair_auroc("known", "unknown"),
        "known_acc": float(np.mean(proto.predict(z[known]) == y[known])),
    }


# ---------------------------------------------------------------- 임베딩 합치기
def combine_fold(zs, y, rec, split, make_scorer=Prototype, vib=None):
    """여러 임베딩의 점수를 검증 녹화 기준으로 표준화한 뒤 평균·최댓값으로 합친다.

    zs: {모델 이름: 임베딩}. 모델마다 검증 녹화를 뺀 학습 녹화로 점수 함수를 맞추고,
    검증 녹화 점수의 중앙값·사분위 범위로 (점수 - 중앙값) / IQR로 바꿔 크기를 맞춘다.
    τ도 같은 검증 녹화에서 알려진 재질 95%가 통과하는 값으로 정한다 (테스트 구간은 쓰지 않음).
    """
    from scipy.stats import spearmanr
    val_recs = inner_val_recordings(split["train"], y, rec)
    fit = np.isin(rec, split["train"]) & ~np.isin(rec, val_recs)
    val = np.isin(rec, val_recs)
    known = np.isin(rec, split["test_known"])
    unknown = np.isin(rec, split["unknown"])
    test = known | unknown
    y_ood = unknown[test].astype(int)

    test_s, val_s = {}, {}
    for name, z in zs.items():
        scorer = make_scorer().fit(z[fit], y[fit])
        ref = scorer.score(z[val])
        med, iqr = np.median(ref), np.subtract(*np.percentile(ref, [75, 25]))
        val_s[name] = (ref - med) / iqr
        test_s[name] = (scorer.score(z[test]) - med) / iqr

    names = list(zs)
    for agg, fn in [("mean", np.mean), ("max", np.max)]:
        key = f"{agg}({'+'.join(names)})"
        val_s[key] = fn(np.stack([val_s[n] for n in names]), axis=0)
        test_s[key] = fn(np.stack([test_s[n] for n in names]), axis=0)

    rows = []
    for name, s in test_s.items():
        tau = fit_tau(val_s[name])
        row = {"method": name, "auroc": auroc(y_ood, s), "fpr95": fpr_at_95_tpr(y_ood, s), "tau": tau,
               "known_pass": float(np.mean(s[y_ood == 0] <= tau)),
               "unknown_reject": float(np.mean(s[y_ood == 1] > tau))}
        if vib is not None:
            row["rho_vib"] = float(spearmanr(s, vib[test])[0])
        rows.append(row)
    return rows
