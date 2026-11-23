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


def per_recording_accuracy(y, pred, rec):
    """녹화별 정확도. 어떤 녹화(=fold)가 특히 어려운지 보는 용도."""
    df = pd.DataFrame({"rec": rec, "true": y, "correct": np.asarray(y) == np.asarray(pred)})
    return (df.groupby("rec").agg(surface=("true", "first"), n=("correct", "size"),
                                  accuracy=("correct", "mean"))
              .sort_values("accuracy"))
