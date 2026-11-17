"""6채널(가속도 3 + 자이로 3) 구성, 정규화, 증강."""


def to_six_channels(recording):
    raise NotImplementedError


def normalize(x, stats=None):
    """train fold 통계로 정규화한다 (test 정보 누수 금지)."""
    raise NotImplementedError


def augment(x):
    raise NotImplementedError
