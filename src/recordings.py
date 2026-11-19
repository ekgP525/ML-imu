"""녹화 21개 복원, 8종 클래스 통합, 6채널 선택.

사용법:
    from recordings import load_dataset
    X, y, rec = load_dataset("data")          # X: (7626, 128, 6), y: 8종 문자열, rec: 0~20
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_RECORDINGS = 21
N_CLASSES = 8

CLASSES = ["carpet", "concrete", "fine_concrete", "hard_tiles",
           "soft_pvc", "soft_tiles", "tiled", "wood"]

MODE = {"carpet": "carpet",
        "soft_pvc": "soft", "soft_tiles": "soft",
        "concrete": "hard", "fine_concrete": "hard", "tiled": "hard",
        "hard_tiles": "hard", "wood": "hard"}

QUAT = slice(0, 4)   # 방향(쿼터니언)
SIX = slice(4, 10)   # 각속도 3 + 선가속도 3
CHANNEL_NAMES = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]

JOIN_THR = 0.01      # 같은 녹화 경계 ≤ 0.0014, 재질이 바뀌는 경계 ≥ 0.10 사이의 값


def load_raw(data_dir="data"):
    """원본 10채널 데이터, 9종 라벨, 80그룹 번호를 그대로 불러온다."""
    d = Path(data_dir)
    X = np.load(d / "X_data.npy").astype(np.float32)
    labels = np.load(d / "label.npy", allow_pickle=True)
    groups = np.load(d / "groups.npy", allow_pickle=True).astype(int)
    if np.issubdtype(labels.dtype, np.number):
        labels = _names_from_details(d / "details.csv", labels)
    return X, labels.astype(str), groups


SURFACES9 = CLASSES + ["hard_tiles_large_space"]


def _names_from_details(path, num_labels):
    """label.npy가 숫자(0~8)면 details.csv의 재질 이름 열로 바꾼다."""
    det = pd.read_csv(path)
    cols = [c for c in det.columns if det[c].astype(str).isin(SURFACES9).all()]
    if not cols:
        raise ValueError(f"details.csv에서 재질 이름 열을 못 찾음: {list(det.columns)}")
    names = det[cols[0]].astype(str).values
    assert len(names) == len(num_labels), "details.csv 행 수가 label.npy와 다름"
    mapping = pd.crosstab(num_labels.astype(int), names)
    assert ((mapping > 0).sum(axis=1) == 1).all(), "숫자 라벨 하나에 재질 이름이 여러 개 대응됨"
    return names


def _qdist(a, b):
    """q와 -q는 같은 방향이므로 둘 중 작은 거리를 쓴다."""
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def check_within_group_order(X, groups):
    """그룹 안에서 파일 순서대로 이어지는지 확인한다(값이 작을수록 정상)."""
    d = [_qdist(X[i, -1, QUAT], X[j, 0, QUAT])
         for g in np.unique(groups)
         for i, j in zip(np.where(groups == g)[0][:-1], np.where(groups == g)[0][1:])]
    return pd.Series(d).describe()


def restore_recordings(X, labels, groups, thr=JOIN_THR, return_boundaries=False):
    """그룹 번호 순서로 이어 붙여, 재질이 같고 경계 쿼터니언 차이가 thr 미만이면 같은 녹화로 본다."""
    recording = np.empty(len(X), dtype=int)
    rows, cur, prev = [], 0, None
    for g in sorted(np.unique(groups)):
        idx = np.where(groups == g)[0]
        lab = labels[idx[0]]
        assert (labels[idx] == lab).all(), f"그룹 {g}에 재질이 섞여 있음"
        if prev is not None:
            p_g, p_lab, p_last = prev
            d = _qdist(X[p_last, -1, QUAT], X[idx[0], 0, QUAT])
            join = (lab == p_lab) and (d < thr)
            rows.append({"from_group": p_g, "to_group": g, "same_surface": lab == p_lab,
                         "qdist": d, "joined": join})
            if not join:
                cur += 1
        recording[idx] = cur
        prev = (g, lab, idx[-1])
    if return_boundaries:
        return recording, pd.DataFrame(rows)
    return recording


def merge_labels(labels):
    """9종 → 8종: hard_tiles_large_space를 hard_tiles로 합친다."""
    return np.where(np.asarray(labels) == "hard_tiles_large_space", "hard_tiles", labels)


def load_dataset(data_dir="data", channels="six", with_groups=False):
    """X, y(8종), recording(0~20)을 반환한다.

    channels="six": 방향 제외 6채널(기본) / "all": 10채널(방향 누수 ablation용)
    with_groups=True면 80그룹 번호도 같이 반환한다(분할 방식 비교용).
    """
    X, labels, groups = load_raw(data_dir)
    recording = restore_recordings(X, labels, groups)
    n = len(np.unique(recording))
    if n != N_RECORDINGS:
        print(f"[경고] 복원된 녹화 수가 {n}개입니다 (기대값 {N_RECORDINGS}). JOIN_THR를 확인하세요.")
    y = merge_labels(labels)
    X = X[:, :, SIX] if channels == "six" else X
    if with_groups:
        return X, y, recording, groups
    return X, y, recording


def summary(y, recording):
    """재질별 구간 수·녹화 수 표 (README 표와 같은지 확인용)."""
    df = pd.DataFrame({"surface": y, "recording": recording})
    out = df.groupby("surface").agg(segments=("recording", "size"),
                                    recordings=("recording", "nunique"))
    out["mode"] = out.index.map(MODE)
    return out


def continuous_chunks(X, labels, groups, thr=JOIN_THR):
    """시간적으로 끊김 없이 이어지는 구간 묶음 번호.
    녹화 경계뿐 아니라 그룹 안에서 쿼터니언이 튀는 곳(예: 그룹 62)에서도 새 묶음을 시작한다.
    64스텝 겹침 분할처럼 구간을 이어 붙이는 작업에는 recording 대신 이 번호를 쓴다."""
    chunk = np.empty(len(X), dtype=int)
    cur, prev_i, prev_lab = 0, None, None
    for g in sorted(np.unique(groups)):
        for i in np.where(groups == g)[0]:
            if prev_i is not None:
                d = _qdist(X[prev_i, -1, QUAT], X[i, 0, QUAT])
                if labels[i] != prev_lab or d >= thr:
                    cur += 1
            chunk[i] = cur
            prev_i, prev_lab = i, labels[i]
    return chunk


def load_chunks(data_dir="data"):
    """continuous_chunks를 원본 데이터에서 바로 계산한다 (load_dataset과 같은 순서)."""
    X, labels, groups = load_raw(data_dir)
    return continuous_chunks(X, labels, groups)
