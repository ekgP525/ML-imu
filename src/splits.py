"""녹화 하나 빼기(leave-one-recording-out) 21-fold 분할."""
import json
from pathlib import Path

FOLDS_PATH = Path(__file__).resolve().parent.parent / "splits" / "folds.json"


def make_folds(recording_ids):
    """녹화마다 하나씩 test로 빼는 fold 목록을 만든다."""
    return [
        {"fold": i, "test": [rid], "train": [r for r in recording_ids if r != rid]}
        for i, rid in enumerate(recording_ids)
    ]


def save_folds(folds, path=FOLDS_PATH):
    with open(path, "w") as f:
        json.dump({"n_folds": len(folds), "folds": folds}, f, indent=2)


def load_folds(path=FOLDS_PATH):
    with open(path) as f:
        return json.load(f)["folds"]
