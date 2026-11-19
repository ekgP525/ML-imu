"""분할 정의.

- 녹화 하나 빼기(leave-one-recording-out) 21-fold: 모든 실험의 기본 분할
- 재질 하나 빼기(leave-one-material-out): D(미등록 판별)·E(신규 등록)용
- 무작위 / 80그룹 분할: 분할 방식 3단계 비교(A 분석)용, 파일로 저장하지 않음
"""
import json
from pathlib import Path

import numpy as np

FOLDS_PATH = Path(__file__).resolve().parent.parent / "splits" / "folds.json"
SEED = 42


# ---------- 기본: 녹화 하나 빼기 ----------
def make_folds(recording_ids):
    """녹화마다 하나씩 test로 빼는 fold 목록을 만든다."""
    recording_ids = sorted(int(r) for r in recording_ids)
    return [
        {"fold": i, "test": [rid], "train": [r for r in recording_ids if r != rid]}
        for i, rid in enumerate(recording_ids)
    ]


def fold_indices(fold, recording):
    """fold 정의(녹화 번호) → 구간 인덱스(train_idx, test_idx)."""
    recording = np.asarray(recording)
    train_idx = np.where(np.isin(recording, fold["train"]))[0]
    test_idx = np.where(np.isin(recording, fold["test"]))[0]
    return train_idx, test_idx


def inner_val_recordings(train_recordings, y, recording):
    """학습 녹화 안에서 검증 녹화를 고른다(τ, λ, 하이퍼파라미터 결정용).
    학습 녹화가 2개 이상 남은 재질마다 번호가 가장 큰 녹화 1개를 검증으로 뗀다."""
    y, recording = np.asarray(y), np.asarray(recording)
    rec_surface = {int(r): y[recording == r][0] for r in train_recordings}
    val = []
    for s in sorted(set(rec_surface.values())):
        recs = sorted(r for r, v in rec_surface.items() if v == s)
        if len(recs) >= 2:
            val.append(recs[-1])
    return val


# ---------- D·E용: 재질 하나 빼기 ----------
def make_material_out_folds(y, recording):
    """재질 m을 학습에서 통째로 빼고, 남은 재질의 녹화를 하나씩 test(알려진 재질)로 뗀다.
    unknown(빠진 재질의 모든 녹화)은 미등록 판별 평가에 쓴다."""
    y, recording = np.asarray(y), np.asarray(recording)
    rec_surface = {int(r): y[recording == r][0] for r in np.unique(recording)}
    folds = []
    for m in sorted(set(rec_surface.values())):
        unknown = sorted(r for r, s in rec_surface.items() if s == m)
        known = sorted(r for r, s in rec_surface.items() if s != m)
        for i, t in enumerate(known):
            folds.append({"held_out": m, "fold": i, "test_known": [t],
                          "unknown": unknown, "train": [r for r in known if r != t]})
    return folds


# ---------- 분할 방식 비교용(A 분석) ----------
def random_folds(y, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def group_folds(groups, n_splits=5):
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits).split(np.zeros(len(groups)), groups=groups))


# ---------- 저장/불러오기 ----------
def save_folds(folds, path=FOLDS_PATH, material_folds=None):
    data = {"n_folds": len(folds), "folds": folds}
    if material_folds is not None:
        data["material_out"] = material_folds
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_folds(path=FOLDS_PATH):
    with open(path) as f:
        return json.load(f)["folds"]


def load_material_folds(path=FOLDS_PATH):
    with open(path) as f:
        return json.load(f)["material_out"]
