"""녹화 21개 복원, 8종 클래스 통합."""

N_RECORDINGS = 21
N_CLASSES = 8


def load_recordings(data_dir="data"):
    """원본 데이터에서 녹화 21개를 복원해 반환한다."""
    raise NotImplementedError


def merge_labels(labels):
    """원본 라벨을 8종으로 통합한다."""
    raise NotImplementedError
