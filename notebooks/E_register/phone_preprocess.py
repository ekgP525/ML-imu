"""E. 스마트폰 녹화 전처리: 앱 CSV → 공개 데이터와 같은 형태의 구간 (N, 128, 6)

입력 폴더 구조 (드라이브 phone_raw/, 녹화 15개)
  phone_raw/
    rug__livingroom__1/          ← 녹화 하나 = 폴더 하나 (또는 앱이 내보낸 .zip 그대로)
      Accelerometer.csv
      Gyroscope.csv
    rug__library__2.zip
    ...
  이름 규칙: {바닥}__{장소}__{회차}   (구분자는 밑줄 두 개)

지원 형식 (열 이름으로 자동 인식)
  - phyphox      : "Time (s)", "Acceleration x (m/s^2)" ... / "Gyroscope x (rad/s)" ...
  - Sensor Logger: time(ns), seconds_elapsed, z, y, x  (TotalAcceleration.csv가 있으면 그걸 우선 사용)
  - 가속도·자이로가 한 CSV에 같이 있는 형식도 열 이름에 acc / gyro가 들어 있으면 인식

처리 순서
  1. 가속도·자이로 스트림 읽기 (시간 단위 ns·ms → s 자동 변환)
  2. 녹화 앞뒤 --trim 초 제거 (카트 출발·정지, 폰 조작 구간)
  3. 두 스트림이 겹치는 시간대에서 --hz 로 리샘플링 (선형 보간)
  4. 128스텝씩 겹치지 않게 자르기 (공개 데이터와 동일). 남는 꼬리는 버림
  5. 채널 순서 gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z (src/recordings.CHANNEL_NAMES와 동일)
  정규화는 여기서 하지 않는다 → 실험 코드에서 학습 녹화만으로 preprocess.fit_stats()

출력 (--out 폴더)
  phone_X.npy     (N, 128, 6) float32
  phone_meta.csv  구간별 material, recording(0~14 정수), recording_name, place, take, order, is_new
  phone_summary.csv  녹화별 실제 샘플링 주기, 길이, 구간 수, 가속도 크기, 사용한 파일 (점검용)

실행
  python notebooks/E_register/phone_preprocess.py --raw phone_raw --new rug,mat
  python notebooks/E_register/phone_preprocess.py --demo        # 가짜 녹화로 점검
"""
import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WIN = 128
CHANNELS = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]

# 가속도 파일 우선순위: 중력 포함 값을 우선 (녹화마다 같은 종류를 써야 함)
ACC_PRIORITY = ["totalacceleration", "acceleration with g", "accelerometer", "acceleration", "acc"]
SKIP_WORDS = ["uncalibrated", "gravity", "magnet", "orientation", "location", "metadata", "device"]


# ───────────────────────────── 파일 읽기 ─────────────────────────────
def list_csvs(path):
    """폴더 또는 zip 안의 CSV를 {파일 이름: DataFrame}으로 읽는다."""
    out = {}
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith(".csv") and not n.startswith("__MACOSX"):
                    out[Path(n).name] = _read_csv(io.BytesIO(z.read(n)))
    else:
        for p in sorted(path.rglob("*.csv")):
            out[p.name] = _read_csv(p)
    return out


def _read_csv(src):
    df = pd.read_csv(src, sep=None, engine="python")       # 쉼표·탭·세미콜론 자동
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _clean(col):
    return re.sub(r"\(.*?\)", " ", col.lower()).strip()


def time_column(df):
    cols = {c: _clean(c) for c in df.columns}
    for c, n in cols.items():
        if n == "seconds_elapsed":
            return c
    for c, n in cols.items():
        if "time" in n:
            return c
    raise ValueError(f"시간 열을 못 찾음: {list(df.columns)}")


def axis_columns(df, keyword=None):
    """x, y, z 축 열을 찾는다. keyword가 있으면 그 단어가 들어간 열 중에서."""
    found = {}
    for c in df.columns:
        n = _clean(c)
        if keyword and keyword not in n:
            continue
        m = re.search(r"(?:^|[\s_])([xyz])$", n)
        if m and "abs" not in n:
            found.setdefault(m.group(1), c)
    if set(found) != {"x", "y", "z"}:
        return None
    return [found["x"], found["y"], found["z"]]


def pick_streams(csvs):
    """{파일명: df} → (acc (t, xyz), gyro (t, xyz), 사용한 파일 설명)."""
    acc, gyro = None, None
    named = [(name, _clean(Path(name).stem), df) for name, df in csvs.items()]
    named = [x for x in named if not any(w in x[1] for w in SKIP_WORDS)]

    # 1) 가속도·자이로가 따로 있는 경우 (파일 이름으로 구분)
    for name, stem, df in named:
        if "gyro" in stem and gyro is None and axis_columns(df):
            gyro = (name, df, axis_columns(df))
    # 중력 제외(linear) 가속도는 다른 가속도 파일이 없을 때만 사용
    by_linear = sorted(named, key=lambda x: "linear" in x[1])
    for key in ACC_PRIORITY:
        for name, stem, df in by_linear:
            if acc is None and key in stem and "gyro" not in stem and axis_columns(df):
                acc = (name, df, axis_columns(df))
    # 2) 한 파일에 같이 있는 경우 (열 이름으로 구분)
    if acc is None or gyro is None:
        for name, stem, df in named:
            a, g = axis_columns(df, "acc"), axis_columns(df, "gyro")
            if a and g:
                acc, gyro = (name, df, a), (name, df, g)
                break
    if acc is None or gyro is None:
        raise ValueError(f"가속도/자이로 CSV를 못 찾음: {list(csvs)}")

    def stream(item):
        name, df, cols = item
        df = df.dropna(subset=cols + [time_column(df)])
        t = df[time_column(df)].to_numpy()
        order = np.argsort(t, kind="stable")
        return t[order].astype(float), df[cols].to_numpy(float)[order]

    (ta, xa), (tg, xg) = stream(acc), stream(gyro)
    # 두 스트림의 시간 기준을 같게: 같은 단위로 바꾼 뒤 공통 시작점을 0으로
    ta_s, tg_s = _unit(ta), _unit(tg)
    t0 = max(ta_s[0], tg_s[0]) if abs(ta_s[0] - tg_s[0]) < 60 else None
    if t0 is None:    # 기준이 다르면 각자 0부터 (보통 같은 앱이면 생기지 않음)
        ta_s, tg_s, t0 = ta_s - ta_s[0], tg_s - tg_s[0], 0.0
    return (ta_s - t0, xa), (tg_s - t0, xg), f"acc={acc[0]}:{acc[2]}, gyro={gyro[0]}:{gyro[2]}"


def _unit(t):
    dt = np.median(np.diff(t))
    return t / 1e9 if dt > 1e5 else (t / 1e3 if dt > 0.5 else t.copy())


# ───────────────────────────── 녹화 하나 처리 ─────────────────────────────
def process_recording(path, hz=100.0, trim=2.0, win=WIN, stride=WIN):
    (ta, xa), (tg, xg), used = pick_streams(list_csvs(path))
    start, end = max(ta[0], tg[0]) + trim, min(ta[-1], tg[-1]) - trim
    if end - start < win / hz:
        raise ValueError(f"{Path(path).name}: 앞뒤 {trim}s를 빼면 구간 하나({win / hz:.2f}s)도 안 나옴")
    t = np.arange(start, end, 1.0 / hz)
    gyro = np.stack([np.interp(t, tg, xg[:, i]) for i in range(3)], 1)
    acc = np.stack([np.interp(t, ta, xa[:, i]) for i in range(3)], 1)
    sig = np.concatenate([gyro, acc], 1).astype(np.float32)

    n = (len(sig) - win) // stride + 1
    X = np.stack([sig[i * stride: i * stride + win] for i in range(n)])

    dta, dtg = np.diff(ta), np.diff(tg)
    info = dict(
        acc_hz=1 / np.median(dta), gyro_hz=1 / np.median(dtg),
        max_gap_s=float(max(dta.max(), dtg.max())),
        duration_s=float(end - start), n_segments=n,
        acc_norm_mean=float(np.linalg.norm(acc, axis=1).mean()),
        used=used,
    )
    return X, info


def parse_name(name):
    parts = re.sub(r"\.zip$", "", name).split("__")
    if len(parts) != 3:
        raise ValueError(f"이름 규칙 {{바닥}}__{{장소}}__{{회차}} 아님: {name}")
    return dict(zip(["material", "place", "take"], parts))


def run(raw_dir, new_floors, out_dir, hz=100.0, trim=2.0, stride=WIN):
    raw_dir = Path(raw_dir)
    items = sorted(p for p in raw_dir.iterdir() if p.is_dir() or p.suffix == ".zip")
    if not items:
        raise FileNotFoundError(f"{raw_dir}에 녹화 폴더/zip이 없음")
    Xs, metas, summ = [], [], []
    for rid, p in enumerate(items):
        info_name = parse_name(p.name)
        X, info = process_recording(p, hz=hz, trim=trim, stride=stride)
        Xs.append(X)
        metas.append(pd.DataFrame(dict(
            **{k: [v] * len(X) for k, v in info_name.items()},
            recording=rid, recording_name=p.name, order=np.arange(len(X)),
            is_new=info_name["material"] in new_floors)))
        summ.append(dict(recording=rid, recording_name=p.name, **info_name, **info))

    X = np.concatenate(Xs)
    meta = pd.concat(metas, ignore_index=True)
    summary = pd.DataFrame(summ)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "phone_X.npy", X)
    meta.to_csv(out / "phone_meta.csv", index=False)
    summary.to_csv(out / "phone_summary.csv", index=False)
    check(X, meta, summary, hz, new_floors)
    return X, meta, summary


# ───────────────────────────── 점검 ─────────────────────────────
def check(X, meta, summary, hz, new_floors):
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(f"\nX {X.shape}  (구간 길이 {WIN / hz:.2f}s @ {hz:g}Hz)")
    tab = meta.groupby(["material", "is_new"]).agg(segments=("order", "size"),
                                                   recordings=("recording", "nunique"))
    print(tab.to_string())
    print("\n" + summary[["recording_name", "acc_hz", "gyro_hz", "max_gap_s", "duration_s",
                          "n_segments", "acc_norm_mean"]].round(2).to_string(index=False))

    warn = []
    if (summary[["acc_hz", "gyro_hz"]].min(axis=1) < hz * 0.95).any():
        warn.append(f"원래 샘플링이 --hz({hz:g})보다 낮은 녹화가 있음 → 업샘플링됨. --hz를 낮추세요")
    if (summary["max_gap_s"] > 0.1).any():
        warn.append("0.1s 넘게 끊긴 녹화가 있음 (앱이 백그라운드로 갔을 수 있음) → max_gap_s 확인")
    g = summary["acc_norm_mean"]
    if g.max() - g.min() > 3:
        warn.append("녹화마다 가속도 크기가 크게 다름 → 중력 포함/제외 파일이 섞였는지 used 열 확인")
    elif g.mean() < 2:
        warn.append(f"가속도 크기 평균 {g.mean():.2f} → 단위가 g이거나 중력 제외 값. 녹화끼리만 같으면 괜찮음")
    if summary["used"].str.replace(r"\[.*?\]", "", regex=True).nunique() > 1:
        warn.append("녹화마다 읽은 파일 종류가 다름 → phone_summary.csv의 used 열 확인")
    missing = set(new_floors) - set(meta["material"])
    if missing:
        warn.append(f"--new에 적은 바닥이 데이터에 없음: {missing}")
    per = meta.groupby("material")["recording"].nunique()
    if (per < 3).any():
        warn.append(f"녹화가 3회 미만인 바닥: {per[per < 3].to_dict()}")
    print("\n" + ("\n".join("[경고] " + w for w in warn) if warn else "[OK] 점검 통과"))


# ───────────────────────── 가짜 녹화 (점검용) ─────────────────────────
def make_demo(raw_dir, seed=42):
    """phyphox 폴더형과 Sensor Logger zip형을 섞어서 15개 녹화를 만든다."""
    rng = np.random.default_rng(seed)
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    floors = {"wood": 0.3, "tile": 0.6, "carpet": 0.15, "rug": 0.1, "mat": 0.2}
    places = ["home", "library", "lab"]
    for fi, (floor, amp) in enumerate(floors.items()):
        for k, place in enumerate(places):
            dur, fs_a, fs_g = rng.uniform(40, 70), rng.uniform(98, 103), rng.uniform(98, 103)
            ta, tg = np.arange(0, dur, 1 / fs_a), np.arange(0.003, dur, 1 / fs_g)
            acc = rng.normal(0, amp, (len(ta), 3)) + [0, 0, 9.81]
            gyr = rng.normal(0, amp / 5, (len(tg), 3))
            name = f"{floor}__{place}__{k + 1}"
            if (fi + k) % 2 == 0:      # phyphox: 폴더, 초 단위
                d = raw_dir / name; d.mkdir(exist_ok=True)
                pd.DataFrame({"Time (s)": ta, "Acceleration x (m/s^2)": acc[:, 0],
                              "Acceleration y (m/s^2)": acc[:, 1], "Acceleration z (m/s^2)": acc[:, 2],
                              "Absolute acceleration (m/s^2)": np.linalg.norm(acc, axis=1)}
                             ).to_csv(d / "Accelerometer.csv", index=False)
                pd.DataFrame({"Time (s)": tg, "Gyroscope x (rad/s)": gyr[:, 0],
                              "Gyroscope y (rad/s)": gyr[:, 1], "Gyroscope z (rad/s)": gyr[:, 2]}
                             ).to_csv(d / "Gyroscope.csv", index=False)
            else:                      # Sensor Logger: zip, 나노초 절대시각 + z,y,x 순서
                t0 = 1.7e18
                with zipfile.ZipFile(raw_dir / f"{name}.zip", "w") as z:
                    for fname, t, v in [("TotalAcceleration.csv", ta, acc), ("Gyroscope.csv", tg, gyr),
                                        ("Accelerometer.csv", ta, acc - [0, 0, 9.81])]:
                        df = pd.DataFrame({"time": (t0 + t * 1e9).astype("int64"), "seconds_elapsed": t,
                                           "z": v[:, 2], "y": v[:, 1], "x": v[:, 0]})
                        z.writestr(fname, df.to_csv(index=False))
    return ["rug", "mat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "phone_raw"))
    ap.add_argument("--new", default="", help="새 바닥 2종, 쉼표로 구분 (예: rug,mat)")
    ap.add_argument("--out", default=str(ROOT / "data" / "phone"))
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--trim", type=float, default=2.0, help="녹화 앞뒤로 버릴 초")
    ap.add_argument("--stride", type=int, default=WIN, help="128이면 겹침 없음 (공개 데이터와 동일)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        a.raw, a.out = "/tmp/phone_demo/raw", "/tmp/phone_demo/out"
        a.new = ",".join(make_demo(a.raw))
    new = [s.strip() for s in a.new.split(",") if s.strip()]
    if len(new) != 2:
        print(f"[경고] 새 바닥은 2종이어야 하는데 {new}")
    run(a.raw, new, a.out, hz=a.hz, trim=a.trim, stride=a.stride)


if __name__ == "__main__":
    main()
