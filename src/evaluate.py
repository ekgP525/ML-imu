"""평가 지표: macro-F1, 모드 macro-F1, AUROC, FPR@95."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def mode_macro_f1(y_true, y_pred, groups):
    """그룹(녹화/구간)별 최빈 예측으로 macro-F1을 계산한다."""
    raise NotImplementedError


def auroc(y_ood, scores):
    """y_ood: OOD=1, ID=0. scores: 클수록 OOD."""
    return roc_auc_score(y_ood, scores)


def fpr_at_95_tpr(y_ood, scores):
    fpr, tpr, _ = roc_curve(y_ood, scores)
    return float(fpr[np.searchsorted(tpr, 0.95)])


def evaluate(y_true, y_pred):
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }
