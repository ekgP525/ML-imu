"""Zenodo에서 원본 데이터를 받아 data/ 아래에 저장한다."""
from pathlib import Path

import requests

ZENODO_RECORD_ID = ""  # TODO: Zenodo 레코드 ID
DATA_DIR = Path(__file__).resolve().parent


def download(record_id: str = ZENODO_RECORD_ID, out_dir: Path = DATA_DIR) -> None:
    if not record_id:
        raise ValueError("ZENODO_RECORD_ID를 설정하세요.")
    meta = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    meta.raise_for_status()
    for f in meta.json()["files"]:
        dest = out_dir / f["key"]
        if dest.exists():
            print(f"skip {dest.name}")
            continue
        print(f"download {dest.name}")
        with requests.get(f["links"]["self"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as fp:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fp.write(chunk)


if __name__ == "__main__":
    download()
