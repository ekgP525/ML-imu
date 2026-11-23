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
