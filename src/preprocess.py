"""6채널 선택, 학습 녹화 기준 정규화, 증강.

정규화 예시:
    stats = fit_stats(X[train_idx])
    X_tr, X_te = normalize(X[train_idx], stats), normalize(X[test_idx], stats)
"""
import numpy as np

SEED = 42


def to_six_channels(x):
    """10채널 → 방향 제외 6채널. 이미 6채널이면 그대로 반환."""
    return x[..., 4:10] if x.shape[-1] == 10 else x


def fit_stats(x_train):
    """채널별 평균·표준편차. 반드시 학습 녹화만 넣는다."""
    flat = x_train.reshape(-1, x_train.shape[-1])
    return {"mean": flat.mean(0), "std": flat.std(0) + 1e-8}


def normalize(x, stats=None):
    """train fold 통계로 정규화한다 (test 정보 누수 금지).
    구간별 정규화는 하지 않는다(진동 크기 정보 보존).
    stats가 None이면 x 자체로 통계를 계산해 (x_norm, stats)를 반환한다 → 학습 세트에만 사용."""
    if stats is None:
        stats = fit_stats(x)
        return (x - stats["mean"]) / stats["std"], stats
    return (x - stats["mean"]) / stats["std"]


def augment(x, y, noise_sd=0.01, scale=0.10, max_shift=8, n_copies=1, seed=SEED):
    """학습 녹화에만 적용: 크기 조절(±scale), 시간 이동(±max_shift 스텝), 가우시안 잡음.
    원본 + n_copies개 증강본을 합쳐 반환한다."""
    rng = np.random.default_rng(seed)
    xs, ys = [x], [y]
    for _ in range(n_copies):
        a = x * rng.uniform(1 - scale, 1 + scale, size=(len(x), 1, 1))
        shifts = rng.integers(-max_shift, max_shift + 1, size=len(x))
        a = np.stack([np.roll(s, k, axis=0) for s, k in zip(a, shifts)])
        a = a + rng.normal(0, noise_sd, size=a.shape)
        xs.append(a.astype(np.float32))
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def overlap_windows(x, y, recording, win=128, stride=64):
    """64스텝 겹침 분할: 같은 녹화의 구간을 원래 순서(인덱스 순)대로 이어 붙인 뒤 다시 자른다.
    학습 세트만 넣는다. x, y, recording은 같은 순서."""
    xs, ys, rs = [], [], []
    recording = np.asarray(recording)
    for r in np.unique(recording):
        idx = np.where(recording == r)[0]          # 인덱스 순 = 시간 순 (녹화 복원과 같은 가정)
        seq = np.concatenate(x[idx], axis=0)
        for s in range(0, len(seq) - win + 1, stride):
            xs.append(seq[s:s + win])
            ys.append(y[idx[0]])
            rs.append(r)
    return np.stack(xs), np.array(ys), np.array(rs)


def trim_edges(x, y, recording, n=2):
    """녹화 앞뒤 n구간(출발·정지로 추정)을 뺀다. ablation용. 남는 인덱스도 반환."""
    recording = np.asarray(recording)
    keep = np.ones(len(x), dtype=bool)
    for r in np.unique(recording):
        idx = np.where(recording == r)[0]
        keep[idx[:n]] = False
        keep[idx[-n:]] = False
    return x[keep], y[keep], recording[keep], np.where(keep)[0]
