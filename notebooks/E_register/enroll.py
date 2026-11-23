"""E. 신규 등록 · 기존 재질 성능 유지 · 모드 승계 (제안서 목표 3, 4장 ③, 5장 평가표)

절차 (splits/folds.json의 material_out 147개 = D와 같은 분할)
  1. 빠진 재질 m의 split마다 emb_{모델}_{m}_{fold}.npy를 불러온다.
  2. 기존 재질 prototype μ_c = split["train"] 녹화 구간 embedding의 평균. 분류는 d(z) = min_c ||z − μ_c||.
  3. m의 녹화(unknown) 하나를 등록 녹화로 정하고, 시간 순으로 구간이 k개 쌓이면 평균을 μ_new로 추가.
     --gate: D와 같은 방식(학습 녹화 안 검증 녹화, 95% 통과)으로 τ를 정해 d(z) > τ인 구간만 쌓는다.
  4. 평가 (모든 split 예측을 모아서 계산)
     - new_acc              : m의 "다른 녹화" 구간이 m으로 분류되는 비율
     - old_macro_f1_before  : 등록 전, 기존 재질 테스트 녹화(test_known) macro-F1
     - old_macro_f1_after   : 등록 후 같은 구간의 macro-F1 (기존 7종 라벨 기준)
     - old_macro_f1_retrain : 8종 전체 학습 embedding(emb_{모델}_none_{21-fold})으로 같은 구간 macro-F1 (상한)
     - old_to_new_rate      : 기존 재질 구간이 새 재질로 빨려 들어간 비율
     - inherit_correct      : μ_new와 가장 가까운 기존 재질의 모드를 물려받았을 때 m의 실제 모드와 맞는지
     - mode_macro_f1 / acc  : 등록 후 전체 테스트 구간의 청소 모드 성능 ("전부 딱딱한 바닥" 정확도 66.3%와 비교)

실행 (레포 루트에서)
  python notebooks/E_register/enroll.py --demo                        # 가짜 embedding으로 점검
  python notebooks/E_register/enroll.py --emb_dir embeddings --model featmlp
  python notebooks/E_register/enroll.py --emb_dir embeddings --model featmlp --gate --log
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))
from recordings import MODE                                              # noqa: E402
from splits import inner_val_recordings, load_folds, load_material_folds  # noqa: E402

SEED = 42
KS = [1, 5, 10, 20]
TPR = 0.95


# ───────────────────── prototype 분류기 (유클리드 거리) ─────────────────────
def prototypes(Z, y):
    classes = sorted(set(y))
    return classes, np.stack([Z[y == c].mean(0) for c in classes])


def dists(Z, P):
    d2 = (Z ** 2).sum(1, keepdims=True) - 2 * Z @ P.T + (P ** 2).sum(1)
    return np.sqrt(np.maximum(d2, 0))


def nearest(Z, classes, P):
    D = dists(Z, P)
    return np.asarray(classes)[D.argmin(1)], D.min(1)


def to_mode(labels, extra=None):
    m = dict(MODE, **(extra or {}))
    return np.array([m[c] for c in labels])


def fit_tau(Z, y, rec, train_recs):
    """D(ood.py)와 같은 규칙: 학습 녹화에서 검증 녹화를 떼고 prototype을 맞춘 뒤,
    검증 구간의 95%가 통과하는 거리."""
    val_recs = inner_val_recordings(train_recs, y, rec)
    fit = np.isin(rec, train_recs) & ~np.isin(rec, val_recs)
    val = np.isin(rec, val_recs)
    classes, P = prototypes(Z[fit], y[fit])
    return float(np.quantile(nearest(Z[val], classes, P)[1], TPR))


# ─────────────────────────── 등록 구간 고르기 ───────────────────────────
def pick_support(sup_idx, k, start, sampling, rng, flagged=None):
    """sup_idx: 등록 녹화 구간 인덱스(시간 순 = 데이터셋 인덱스 순).
    contiguous: start부터 시간 순으로 k개 (제안서: '같은 녹화에서 k개 쌓이면')
    random    : 녹화 안에서 무작위 k개 (비교용)
    flagged   : 미등록 판정 여부. 주어지면 판정된 구간만 쌓고, k개가 안 모이면 등록 실패(None)"""
    if sampling == "random":
        pool = sup_idx if flagged is None else sup_idx[flagged]
        return None if len(pool) < k else rng.choice(pool, k, replace=False)
    seq = sup_idx[start:]
    if flagged is not None:
        seq = seq[flagged[start:]]
    return None if len(seq) < k else seq[:k]


# ─────────────────────────────── 실험 ───────────────────────────────
def run(y, rec, load, model, splits, folds21, ks=KS, n_repeats=10, sampling="contiguous", gate=False):
    """load(held, fold) -> embedding (N, D). held="none"이면 fold는 21-fold 번호."""
    y, rec = np.asarray(y), np.asarray(rec)
    fold_of_test = {f["test"][0]: f["fold"] for f in folds21}   # 녹화 번호 → 21-fold 번호
    rows, base_rows = [], []

    for held in sorted({s["held_out"] for s in splits}):
        my_splits = [s for s in splits if s["held_out"] == held]
        held_recs = my_splits[0]["unknown"]
        if len(held_recs) < 2:
            print(f"[skip] {held}: 녹화 {len(held_recs)}개 → 등록/평가 녹화를 나눌 수 없음")
            continue

        cache = []
        for s in my_splits:
            Z = load(held, s["fold"])
            tr = np.isin(rec, s["train"])
            te = np.where(np.isin(rec, s["test_known"]))[0]
            classes, P = prototypes(Z[tr], y[tr])
            tau = fit_tau(Z, y, rec, s["train"]) if gate else None
            cache.append(dict(fold=s["fold"], Z=Z, te=te, classes=classes, P=P, tau=tau,
                              test_rec=s["test_known"][0]))

        # ── 등록 전 / 전체 재학습(상한) / 모드 기준선 ──
        yt, yp, yp_rt, has_retrain = [], [], [], True
        for c in cache:
            yt.append(y[c["te"]])
            yp.append(nearest(c["Z"][c["te"]], c["classes"], c["P"])[0])
            try:
                f21 = fold_of_test[c["test_rec"]]
                Z8 = load("none", f21)
                tr8 = ~np.isin(rec, [c["test_rec"]])
                cl8, P8 = prototypes(Z8[tr8], y[tr8])
                yp_rt.append(nearest(Z8[c["te"]], cl8, P8)[0])
            except FileNotFoundError:
                has_retrain = False
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        old_labels = sorted(set(yt))
        true_mode_all = np.concatenate([to_mode(yt), to_mode(y[np.isin(rec, held_recs)])])
        base_rows.append(dict(
            model=model, held=held, true_mode=MODE[held], n_splits=len(cache),
            old_macro_f1_before=f1_score(yt, yp, labels=old_labels, average="macro"),
            old_macro_f1_retrain=(f1_score(yt, np.concatenate(yp_rt), labels=old_labels, average="macro")
                                  if has_retrain else np.nan),
            mode_acc_allhard=(true_mode_all == "hard").mean(),
        ))

        # ── 등록 후 ──
        for ei, er in enumerate(held_recs):
            sup_idx = np.where(rec == er)[0]                   # 인덱스 순 = 시간 순
            new_te = np.where(np.isin(rec, [r for r in held_recs if r != er]))[0]
            for k in ks:
                for r in range(n_repeats):
                    rng = np.random.default_rng([SEED, ei, k, r])
                    start = int(rng.integers(0, max(1, len(sup_idx) - k + 1)))
                    shared = pick_support(sup_idx, k, start, "random", rng) if sampling == "random" else None
                    acc = dict(old_t=[], old_p=[], new_hit=[], all_t=[], all_p=[], mt=[], mp=[], inherit=[])
                    for c in cache:
                        Z, te, classes, P = c["Z"], c["te"], c["classes"], c["P"]
                        if c["tau"] is not None:
                            flagged = nearest(Z[sup_idx], classes, P)[1] > c["tau"]
                            sup = pick_support(sup_idx, k, start, sampling, rng, flagged=flagged)
                        else:
                            sup = shared if shared is not None else pick_support(sup_idx, k, start, sampling, rng)
                        if sup is None:          # 미등록 판정 구간이 k개 안 모임 → 등록 실패
                            continue
                        mu_new = Z[sup].mean(0, keepdims=True)
                        nearest_old = classes[dists(mu_new, P).argmin()]
                        inherited = MODE[nearest_old]
                        acc["inherit"].append((nearest_old, inherited == MODE[held]))
                        cl2, P2 = classes + [held], np.vstack([P, mu_new])
                        p_old = nearest(Z[te], cl2, P2)[0]
                        p_new = nearest(Z[new_te], cl2, P2)[0]
                        acc["old_t"].append(y[te]); acc["old_p"].append(p_old)
                        acc["new_hit"].append(p_new == held)
                        t_all, p_all = np.concatenate([y[te], y[new_te]]), np.concatenate([p_old, p_new])
                        acc["all_t"].append(t_all); acc["all_p"].append(p_all)
                        acc["mt"].append(to_mode(t_all)); acc["mp"].append(to_mode(p_all, {held: inherited}))

                    row = dict(model=model, held=held, enroll_rec=er, k=k, repeat=r,
                               enroll_success=len(acc["inherit"]) / len(cache))
                    if acc["inherit"]:
                        cat = lambda key: np.concatenate(acc[key])  # noqa: E731
                        old_t, old_p = cat("old_t"), cat("old_p")
                        all_t, all_p, mt, mp = cat("all_t"), cat("all_p"), cat("mt"), cat("mp")
                        names = [n for n, _ in acc["inherit"]]
                        row.update(
                            new_acc=cat("new_hit").mean(),
                            old_macro_f1_after=f1_score(old_t, old_p, labels=sorted(set(old_t)), average="macro"),
                            old_to_new_rate=(old_p == held).mean(),
                            all_macro_f1=f1_score(all_t, all_p, average="macro"),
                            all_acc=(all_t == all_p).mean(),
                            inherit_correct=np.mean([ok for _, ok in acc["inherit"]]),
                            nearest_old=max(set(names), key=names.count),
                            mode_macro_f1=f1_score(mt, mp, labels=["carpet", "soft", "hard"],
                                                   average="macro", zero_division=0),
                            mode_acc=(mt == mp).mean(),
                        )
                    rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(base_rows)


METRICS = ["enroll_success", "new_acc", "old_macro_f1_after", "old_to_new_rate",
           "all_macro_f1", "all_acc", "inherit_correct", "mode_macro_f1", "mode_acc"]


def summarize(df, base):
    g = df.groupby(["model", "held", "k"])
    by_held = g[METRICS].mean().reset_index()
    by_held["nearest_old"] = g["nearest_old"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None).values
    by_held = by_held.merge(base, on=["model", "held"])
    by_k = by_held.groupby(["model", "k"])[METRICS + ["old_macro_f1_before", "old_macro_f1_retrain"]] \
                  .agg(["mean", "std"])
    return by_held, by_k


# ─────────────────── 가짜 embedding (파이프라인 점검용) ───────────────────
def make_demo(splits, dim=32, seed=SEED):
    """folds.json의 실제 녹화-재질 구성으로 가짜 구간과 embedding을 만든다."""
    rng = np.random.default_rng(seed)
    rec_mat = {r: s["held_out"] for s in splits for r in s["unknown"]}
    group_c = {m: rng.normal(0, 2, dim) for m in ["carpet", "soft", "hard"]}
    centers = {m: group_c[MODE[m]] + rng.normal(0, 1.0, dim) for m in MODE}
    centers["wood"] = 0.5 * centers["wood"] + 0.5 * group_c["soft"]      # wood는 조용한 쪽
    y, rec, X = [], [], []
    for r in sorted(rec_mat):
        off = rng.normal(0, 0.6, dim)
        n = int(rng.integers(150, 500))
        X.append(centers[rec_mat[r]] + off + rng.normal(0, 1.0, (n, dim)))
        y += [rec_mat[r]] * n; rec += [r] * n
    X = np.vstack(X)
    return np.array(y), np.array(rec), (lambda held, fold: X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--emb_dir", default=str(ROOT / "embeddings"))
    ap.add_argument("--model", default="featmlp")
    ap.add_argument("--gate", action="store_true", help="τ로 미등록 판정된 구간만 등록에 사용")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--sampling", choices=["contiguous", "random"], default="contiguous")
    ap.add_argument("--out", default=str(ROOT / "notebooks" / "E_register" / "out"))
    ap.add_argument("--log", action="store_true", help="results/results.csv에 k별 요약 기록")
    a = ap.parse_args()

    splits, folds21 = load_material_folds(), load_folds()
    if a.demo:
        y, rec, load = make_demo(splits)
        a.model = "demo"
    else:
        from recordings import load_dataset
        _, y, rec = load_dataset(a.data_dir)
        emb_dir = Path(a.emb_dir)
        load = lambda held, fold: np.load(emb_dir / f"emb_{a.model}_{held}_{fold}.npy")  # noqa: E731

    df, base = run(y, rec, load, a.model, splits, folds21,
                   n_repeats=a.repeats, sampling=a.sampling, gate=a.gate)
    by_held, by_k = summarize(df, base)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.model}_{a.sampling}{'_gate' if a.gate else ''}"
    df.to_csv(out / f"enroll_raw_{tag}.csv", index=False)
    by_held.to_csv(out / f"enroll_by_held_{tag}.csv", index=False)
    by_k.to_csv(out / f"enroll_by_k_{tag}.csv")

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n[k별 평균 (빠진 재질 8종 평균)]")
    print(by_k.xs("mean", axis=1, level=1).round(3))
    print(f"\n[재질별, k={max(KS)}]")
    cols = ["held", "true_mode", "nearest_old", "inherit_correct", "new_acc",
            "old_macro_f1_before", "old_macro_f1_after", "old_macro_f1_retrain"]
    print(by_held[by_held.k == max(KS)][cols].round(3).to_string(index=False))

    if a.log and not a.demo:
        from evaluate import log_result
        mean_k = by_k.xs("mean", axis=1, level=1).reset_index()
        for _, r in mean_k.iterrows():
            log_result("E", a.model, {"macro_f1": r["all_macro_f1"], "accuracy": r["all_acc"],
                                      "mode_macro_f1": r["mode_macro_f1"]},
                       setting=f"enroll_k{int(r['k'])}{'_gate' if a.gate else ''}", fold="all",
                       note=f"new_acc {r['new_acc']:.3f}, old_f1 {r['old_macro_f1_after']:.3f}, "
                            f"inherit {r['inherit_correct']:.3f}")


if __name__ == "__main__":
    main()
