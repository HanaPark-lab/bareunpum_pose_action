"""9일차 — Train 전용 Keypoint 보강 후 Val 비교 (수업 53·54·55 대응).

보강 방법 (Train 영상의 Keypoint CSV 에만 적용, Val/Test 는 절대 보강하지 않음):
  flip     좌우 반전 (수업 horizontal_flip_dataframe)
  jitter   관절 좌표에 작은 흔들림 (수업 jitter_dataframe, 어깨폭 × ratio)
  speed    시간축 0.8배·1.25배 (같은 동작을 느리게/빠르게)
기본은 위험 클래스 영상만 보강 (데이터가 가장 적은 곳). --all 이면 OTHER 영상도.

실행:  uv run python scripts/d09_augment_compare.py
       uv run python scripts/d09_augment_compare.py --methods flip jitter --jitter 0.02
결과:  reports/day09/augment_compare.csv  (baseline vs 보강, 같은 Val)
       --save 를 주면 보강 모델을 models/ 에 저장 (기본은 저장 안 함 → d06 모델 유지)
확인:  위험 Recall 이 오르면서 OTHER 오경보가 같이 오르지 않는지.
       VICON 3D(FBX) → 여러 각도 2D 투영 보강은 시간이 남을 때의 확장 (Blender 필요).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.augmentation_utils import horizontal_flip_dataframe, jitter_dataframe
from src.common import DATA_DIR, KEYPOINT_DIR, MODEL_DIR, REPORT_DIR, ensure_dir, load_actions, load_clips, load_json
from src.window_utils import build_clip_windows

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d06_train_rf import make_rf, threshold_metrics  # noqa: E402


def to_float(v):
    return None if v in ("", None) else float(v)


def speed_dataframe(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_sec"] = out["timestamp_sec"].astype(float) * factor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=["flip", "jitter", "speed"])
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--all", action="store_true", help="OTHER 영상도 보강")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    cfg = load_json("train.json")
    split_cfg = load_json("split.json")
    actions = load_actions()
    d = np.load(DATA_DIR / "windows" / "windows.npz", allow_pickle=True)
    X, y = d["X"], d["y"].astype(str)
    meta = pd.read_csv(DATA_DIR / "windows" / "windows_meta.csv")
    tr = (meta["split"] == "train").to_numpy() & (~meta["source"].isin(["hmdb51", "vicon2d"])).to_numpy()
    va = (meta["split"] == "val").to_numpy()
    if not va.any():
        raise SystemExit("Val 창이 없습니다. 카톡 영상 라벨·split.json 을 먼저 채우세요.")

    # ① Train 영상만 골라 보강 창 만들기
    aug_X, aug_y = [], []
    for clip in load_clips():
        a = actions.get(clip["action_id"])
        if a is None or a["role"] == "PRACTICE" or clip.get("performer_id") not in split_cfg["train"]:
            continue
        if a["role"] != "DANGER" and not args.all:
            continue
        kp = KEYPOINT_DIR / f"{clip['video']}.csv"
        if not kp.exists():
            continue
        df = pd.read_csv(kp)
        variants = []
        if "flip" in args.methods:
            variants.append(horizontal_flip_dataframe(df))
        if "jitter" in args.methods:
            variants.append(jitter_dataframe(df, args.jitter, cfg["random_state"]))
        if "speed" in args.methods:
            variants += [speed_dataframe(df, 0.8), speed_dataframe(df, 1.25)]
        seg_s, seg_e = to_float(clip["start_sec"]), to_float(clip["end_sec"])
        for k, v in enumerate(variants):
            s_s, s_e = seg_s, seg_e
            if "speed" in args.methods and k >= len(variants) - 2 and seg_s is not None:
                f = 0.8 if k == len(variants) - 2 else 1.25
                s_s, s_e = seg_s * f, seg_e * f
            for vec, label, _, _ in build_clip_windows(v, a["rf_label"], s_s, s_e, cfg):
                aug_X.append(vec)
                aug_y.append(label)
    print(f"[보강] 창 {len(aug_X)}개 추가 (방법 {args.methods})")

    rows = []
    for name, Xt, yt in [("E1 baseline", X[tr], y[tr]),
                         ("E1 + aug", np.vstack([X[tr]] + ([np.stack(aug_X)] if aug_X else [])), np.concatenate([y[tr], np.array(aug_y, dtype=str)]))]:
        m = make_rf(cfg).fit(Xt, yt)
        r = {"exp": name, "train_windows": len(yt)}
        r.update(threshold_metrics(m, X[va], y[va], cfg["danger_threshold"]))
        rows.append(r)
        if name == "E1 + aug" and args.save:
            joblib.dump(m, MODEL_DIR / "rf_action_baseline.joblib")
            info_p = MODEL_DIR / "rf_training_info.json"
            info = json.loads(info_p.read_text(encoding="utf-8"))
            info.update(model_version="bareunpum-E1-aug", experiment="E1+aug", augmentation=args.methods,
                        train_windows=len(yt), classes=[str(c) for c in m.classes_])
            info_p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            print("[저장] 보강 모델로 models/ 교체")
    res = pd.DataFrame(rows)
    out = ensure_dir(REPORT_DIR / "day09")
    res.to_csv(out / "augment_compare.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
