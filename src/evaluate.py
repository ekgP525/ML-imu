"""평가 지표: macro-F1, 정확도, 모드 macro-F1, AUROC, FPR@95%TPR, 결과 기록.

주의: 녹화 하나 빼기는 fold마다 test 재질이 1종뿐이라 fold별 macro-F1은 의미가 없다.
      21개 fold의 예측을 모두 모은 뒤 evaluate()를 한 번 호출한다 (fold="all"로 기록).
"""
import csv
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from recordings import MODE

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "results.csv"
RESULT_COLUMNS = ["date", "owner", "model", "setting", "fold", "macro_f1", "accuracy",
                  "mode_macro_f1", "auroc", "fpr95", "note"]


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def to_mode(labels):
    """재질 → 청소 모드(carpet / soft / hard). 모르는 값(미등록 등)은 'unknown'."""
    return np.array([MODE.get(str(v), "unknown") for v in labels])


def mode_macro_f1(y_true, y_pred):
    """청소 모드 3종(카펫·부드러운·딱딱한) 기준 macro-F1.
    기준선: 전부 '딱딱한 바닥'으로 고르면 정확도 66.3%."""
    return f1_score(to_mode(y_true), to_mode(y_pred), average="macro",
                    labels=["carpet", "soft", "hard"])


def auroc(y_ood, scores):
    """y_ood: OOD=1, ID=0. scores: 클수록 OOD."""
    return roc_auc_score(y_ood, scores)


def fpr_at_95_tpr(y_ood, scores):
    """알려진 재질(ID)을 95% 통과시키는 임계값에서, 미등록(OOD)을 알려진 재질로 잘못 통과시키는 비율.
    scores는 클수록 OOD. ID를 양성으로 보고 계산한다."""
    y_id = 1 - np.asarray(y_ood)
    fpr, tpr, _ = roc_curve(y_id, -np.asarray(scores))
    return float(fpr[min(np.searchsorted(tpr, 0.95), len(fpr) - 1)])


def evaluate(y_true, y_pred):
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "mode_macro_f1": mode_macro_f1(y_true, y_pred),
    }


def log_result(owner, model, metrics, setting="8class", fold="all", note="", path=RESULTS_PATH):
    """results/results.csv에 한 줄 추가한다. metrics는 evaluate() 결과 등 dict."""
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    row = {"date": date.today().isoformat(), "owner": owner, "model": model,
           "setting": setting, "fold": fold, "note": note}
    for k in ["macro_f1", "accuracy", "mode_macro_f1", "auroc", "fpr95"]:
        row[k] = "" if metrics.get(k) is None else round(float(metrics[k]), 4)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
