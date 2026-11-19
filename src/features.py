"""특성 A~C. 입력 x: (N, T, 6) 원본(정규화 전) 6채널.

스케일링은 각자 모델 파이프라인(StandardScaler 등)에서 학습 fold 기준으로 한다.

    from features import extract_features
    F = extract_features(X)              # DataFrame, A+B+C 전부
    F_A = extract_features(X, "A")       # ablation: A만
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

CHANNEL_NAMES = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]


def _names(prefix, stats, ch=CHANNEL_NAMES):
    return [f"{prefix}_{s}_{c}" for s in stats for c in ch]


def features_a(x):
    """A. 진동 세기: 채널별 표준편차, RMS 에너지, 최댓값 − 최솟값."""
    f = np.concatenate([x.std(1), np.sqrt((x ** 2).mean(1)), x.max(1) - x.min(1)], axis=1)
    return pd.DataFrame(f, columns=_names("A", ["std", "rms", "ptp"]))


def features_b(x):
    """B. 분포 형태: 채널별 평균, 왜도, 첨도."""
    f = np.concatenate([x.mean(1), skew(x, axis=1), kurtosis(x, axis=1)], axis=1)
    return pd.DataFrame(np.nan_to_num(f), columns=_names("B", ["mean", "skew", "kurt"]))


def features_c(x):
    """C. 주파수: FFT 파워(직류 성분 제외)를 저·중·고 3대역으로 나눈 에너지."""
    p = np.abs(np.fft.rfft(x - x.mean(1, keepdims=True), axis=1)) ** 2
    p = p[:, 1:, :]
    f = np.concatenate([part.sum(1) for part in np.array_split(p, 3, axis=1)], axis=1)
    return pd.DataFrame(f, columns=_names("C", ["low", "mid", "high"]))


def extract_features(x, groups="ABC"):
    """groups 문자열에 든 특성 그룹만 합쳐 반환한다 (예: "A", "AB", "ABC")."""
    fn = {"A": features_a, "B": features_b, "C": features_c}
    return pd.concat([fn[g](x) for g in groups], axis=1)
