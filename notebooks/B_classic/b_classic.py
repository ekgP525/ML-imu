"""B. 고전 모델 공통 코드: 녹화 하나 빼기 21-fold 루프, 진동 크기 규칙 기준선.

노트북에서:
    import sys; sys.path += ["src", "notebooks/B_classic"]
    from b_classic import RuleBaseline, run_loro
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from splits import fold_indices, load_folds  # noqa: E402

SEED = 42
RULE_FEATURE = "A_std_acc_z"   # 가속도 Z축 표준편차 = 위아래 흔들림 세기


class RuleBaseline:
    """진동 크기 임계값 규칙.

    학습 녹화에서 재질별 log(진동 세기) 중앙값을 구하고, 새 구간은 가장 가까운 중앙값의 재질로 고른다.
    중앙값을 정렬하면 이웃한 두 값의 가운데가 곧 임계값이 된다 (1차원 최근접 중심 = 구간 나누기 규칙).
    """

    def __init__(self, feature=RULE_FEATURE):
        self.feature = feature

    def _v(self, F):
        return np.log(np.asarray(F[self.feature], dtype=float) + 1e-8)

    def fit(self, F, y):
        v, y = self._v(F), np.asarray(y)
        self.classes_ = np.unique(y)
        self.centers_ = np.array([np.median(v[y == c]) for c in self.classes_])
        return self

    def predict(self, F):
        d = np.abs(self._v(F)[:, None] - self.centers_[None])
        return self.classes_[d.argmin(axis=1)]

    def thresholds(self):
        """재질을 진동 세기 순으로 정렬하고, 이웃 재질 사이의 경계값(원래 단위)을 표로 반환한다."""
        order = np.argsort(self.centers_)
        c, v = self.classes_[order], self.centers_[order]
        return pd.DataFrame({"surface": c, "center": np.exp(v),
                             "upper_threshold": np.append(np.exp((v[:-1] + v[1:]) / 2), np.inf)})


def run_loro(make_model, F, y, rec, folds=None):
    """녹화 하나 빼기 21-fold. 각 fold의 test 구간 예측을 데이터셋 순서대로 모아 반환한다.

    make_model: 인자 없이 새 모델을 만드는 함수 (fit/predict 제공)
    F: 특성 DataFrame 또는 배열 (데이터셋 순서), y: 8종 라벨, rec: 녹화 번호
    """
    folds = load_folds() if folds is None else folds
    y = np.asarray(y)
    pred = np.empty(len(y), dtype=object)
    done = np.zeros(len(y), dtype=bool)
    for fold in folds:
        tr, te = fold_indices(fold, rec)
        Ftr = F.iloc[tr] if hasattr(F, "iloc") else F[tr]
        Fte = F.iloc[te] if hasattr(F, "iloc") else F[te]
        pred[te] = make_model().fit(Ftr, y[tr]).predict(Fte)
        done[te] = True
    assert done.all(), "test에 한 번도 안 들어간 구간이 있음 (folds.json과 녹화 번호 확인)"
    return pred.astype(str)


class FeatureMLP:
    """특성(A~C) → 표준화 → MLP. 마지막 은닉층 출력을 임베딩으로, 출력층 직전 값을 logits로 내보낸다.

    표준화는 fit에 들어온 학습 구간으로만 맞춘다 (test 누수 없음).
    """

    def __init__(self, hidden=(64, 32), alpha=1e-3, max_iter=300, seed=SEED):
        self.hidden, self.alpha, self.max_iter, self.seed = hidden, alpha, max_iter, seed

    def fit(self, F, y):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        self.scaler_ = StandardScaler().fit(np.asarray(F, dtype=float))
        self.mlp_ = MLPClassifier(hidden_layer_sizes=self.hidden, activation="relu",
                                  alpha=self.alpha, max_iter=self.max_iter,
                                  early_stopping=True, validation_fraction=0.1,
                                  n_iter_no_change=15, random_state=self.seed)
        # 라벨을 숫자(0..K-1)로 바꿔 학습한다. sklearn 일부 버전은 early_stopping 검증 점수 계산에서
        # 문자열 라벨이면 TypeError(isnan)가 난다. 숫자 i = self.classes_[i] (알파벳 순).
        self.classes_, y_idx = np.unique(np.asarray(y).astype(str), return_inverse=True)
        self.mlp_.fit(self.scaler_.transform(np.asarray(F, dtype=float)), y_idx)
        return self

    def embed(self, F):
        """마지막 은닉층(ReLU 후) 출력. 크기: (N, hidden[-1])."""
        h = self.scaler_.transform(np.asarray(F, dtype=float))
        for W, b in zip(self.mlp_.coefs_[:-1], self.mlp_.intercepts_[:-1]):
            h = np.maximum(h @ W + b, 0)
        return h.astype(np.float32)

    def logits(self, F):
        """softmax 전 출력. 열 순서는 self.classes_ (알파벳 순)."""
        z = self.embed(F) @ self.mlp_.coefs_[-1] + self.mlp_.intercepts_[-1]
        if z.shape[1] == 1:   # 2종일 때 sklearn은 출력 1개 → 2열로 맞춤
            z = np.hstack([np.zeros_like(z), z])
        return z.astype(np.float32)

    def predict(self, F):
        return self.classes_[self.logits(F).argmax(axis=1)]


def select_features(F_all, spec):
    """특성 조합 선택.

    spec 예: "A", "AB", "ABC", "AC"
      - "-통계" : 그 통계 열을 뺌. "ABC-mean"(채널 평균 6개 제외), "AC-rms"(RMS 6개 제외)
      - "log" 접두: 고른 열에 로그를 씌움 (A·C는 모두 양수). "logAC-rms"
        진동이 큰 바닥일수록 값과 거리가 커지는 문제를 줄이려는 용도 (배율 → 차이로)
    """
    use_log = spec.startswith("log")
    groups, _, drop = spec[3:].partition("-") if use_log else spec.partition("-")
    cols = [c for c in F_all.columns if c[0] in groups]
    if drop:
        cols = [c for c in cols if c.split("_")[1] != drop]
    F = F_all[cols]
    if use_log:
        assert not groups.replace("A", "").replace("C", ""), "로그는 A·C(양수 특성)에만"
        F = np.log(F.clip(lower=0) + 1e-8)
    return F


class LogRegEmbed:
    """로지스틱 회귀(L2) + 내보내기용 임베딩·logits.

    임베딩 = 학습 fold 기준으로 표준화한 특성 (로지스틱 회귀가 실제로 보는 공간)
    logits = decision_function (softmax 전 재질별 점수, 열 순서 self.classes_)
    """

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, F, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Fa = np.asarray(F, dtype=float)
        self.scaler_ = StandardScaler().fit(Fa)
        self.clf_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000)
        self.clf_.fit(self.scaler_.transform(Fa), np.asarray(y).astype(str))
        self.classes_ = self.clf_.classes_
        return self

    def embed(self, F):
        return self.scaler_.transform(np.asarray(F, dtype=float)).astype(np.float32)

    def logits(self, F):
        z = self.clf_.decision_function(self.embed(F))
        if z.ndim == 1:
            z = np.stack([-z, z], axis=1)
        return z.astype(np.float32)

    def predict(self, F):
        return self.classes_[self.logits(F).argmax(axis=1)]


def make_classic(name, seed=SEED):
    """수업에서 다룬 고전 모델. 표준화는 파이프라인 안에서 학습 fold로만 맞춘다."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "rule":
        return RuleBaseline()
    if name == "logreg_l2":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    elif name == "logreg_l1":
        # saga는 이 데이터에서 너무 느려(조합당 5분+) liblinear 일대다(one-vs-rest)로 학습한다.
        from sklearn.multiclass import OneVsRestClassifier
        clf = OneVsRestClassifier(LogisticRegression(penalty="l1", C=0.5, solver="liblinear",
                                                     class_weight="balanced", max_iter=3000))
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=15)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=5, class_weight="balanced",
                                     n_jobs=-1, random_state=seed)
    elif name == "mlp":
        return FeatureMLP(seed=seed)
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), clf)


def export_embeddings(F, y, rec, folds, out_dir, held_key="held_out", make_model=FeatureMLP,
                      name="mlp", skip_existing=True):
    """split마다 학습 녹화로 모델을 맞추고, 전체 7626구간의 임베딩·logits를 데이터셋 순서로 저장한다.

    파일: emb_{name}_{빠진재질}_{fold}.npy, logits_{name}_{빠진재질}_{fold}.npy
          (빠진 재질이 없는 21-fold는 'none'), classes_{name}_{빠진재질}_{fold}.json (logits 열 순서)
    """
    import json
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y, rec = np.asarray(y), np.asarray(rec)
    Fa = np.asarray(F, dtype=float)
    for i, split in enumerate(folds):
        held = split.get(held_key, "none")
        tag = f"{name}_{held}_{split['fold']}"
        if skip_existing and (out_dir / f"emb_{tag}.npy").exists():
            continue
        tr = np.isin(rec, split["train"])
        model = make_model().fit(Fa[tr], y[tr])
        np.save(out_dir / f"emb_{tag}.npy", model.embed(Fa))
        np.save(out_dir / f"logits_{tag}.npy", model.logits(Fa))
        with open(out_dir / f"classes_{tag}.json", "w") as f:
            json.dump([str(c) for c in model.classes_], f)
        if (i + 1) % 10 == 0 or i + 1 == len(folds):
            print(f"{i + 1}/{len(folds)} 저장: {tag}")


def per_recording_accuracy(y, pred, rec):
    """녹화별 정확도. 어떤 녹화(=fold)가 특히 어려운지 보는 용도."""
    df = pd.DataFrame({"rec": rec, "true": y, "correct": np.asarray(y) == np.asarray(pred)})
    return (df.groupby("rec").agg(surface=("true", "first"), n=("correct", "size"),
                                  accuracy=("correct", "mean"))
              .sort_values("accuracy"))
