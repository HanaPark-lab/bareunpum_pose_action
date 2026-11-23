"""3일차 — 관절각도·정규화·38-D Frame Feature + 바른품 허리 값(몸통 기울기·무릎 각도).

실행:  uv run python scripts/d03_frame_features.py --video C004_T001
       (C004 안고 물건 줍기, C009 무릎 꿇기, C010 쪼그려 앉기 를 비교해 보세요)
결과:  reports/day03/<video>_features.csv   Frame별 38-D + trunk_lean_deg, knee_left/right_deg
       reports/day03/<video>_waist.png       시간축 그래프 (45°, 160° 기준선)
확인:  ① 38개 Feature가 수업과 같은 순서로 나오는지 (sequence_utils.FRAME_FEATURE_NAMES)
       ② 숙일 때 trunk_lean 이 45°를 넘는지, 그때 무릎이 160°보다 펴져 있는지
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import REPORT_DIR, ensure_dir, load_clips, load_json, video_path
from src.pose_extract import load_pose_model, main_person_rows, select_main_track, track_video
from src.sequence_utils import FRAME_FEATURE_NAMES, frame_feature_from_row
from src.waist_utils import frame_violates, frame_waist_values

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="C004_T001")
    args = ap.parse_args()

    out_dir = ensure_dir(REPORT_DIR / "day03")
    rule = load_json("waist_rule.json")
    clip = next((c for c in load_clips() if c["video"] == args.video), None)
    if clip is None:
        raise SystemExit(f"clips.csv 에 {args.video} 가 없습니다.")

    model, config = load_pose_model()
    frames, fps = track_video(video_path(clip), model, config)
    main_id, summary = select_main_track(frames)
    print(f"[Tracking] 사람 {summary.get('track_count')}명 중 수행자 Track {main_id} 선택")
    rows = main_person_rows(frames, main_id)

    out_rows = []
    for r in rows:
        rec = {"frame_id": r["frame_id"], "timestamp_sec": r["timestamp_sec"], "pose_detected": r["pose_detected"]}
        feat = frame_feature_from_row(pd.Series(r)) if r["pose_detected"] else None
        for i, name in enumerate(FRAME_FEATURE_NAMES):
            rec[name] = round(float(feat[i]), 4) if feat is not None else ""
        if r["pose_detected"]:
            kp = np.array([[r[f"{n}_x_px"], r[f"{n}_y_px"]] for n in _names()], dtype=np.float32)
            cf = np.array([r[f"{n}_conf"] for n in _names()], dtype=np.float32)
            wv = frame_waist_values(kp, cf, rule["min_joint_conf"])
            viol = frame_violates(wv, rule)
        else:
            wv, viol = {"trunk_lean_deg": None, "knee_left_deg": None, "knee_right_deg": None}, None
        rec.update({k: ("" if v is None else round(v, 2)) for k, v in wv.items()})
        rec["waist_violation"] = "" if viol is None else int(viol)
        out_rows.append(rec)

    csv_path = out_dir / f"{args.video}_features.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    df = pd.DataFrame(out_rows).replace("", np.nan)
    n_feat = df[FRAME_FEATURE_NAMES[0]].notna().sum()
    print(f"[Feature] 38-D Feature 생성 Frame {n_feat}/{len(df)}  (열 {len(FRAME_FEATURE_NAMES)}개)")
    v = df["waist_violation"].dropna()
    print(f"[허리] 판정 가능 Frame {len(v)}/{len(df)}  위반 비율 {v.mean() if len(v) else float('nan'):.2f}"
          f"  (규칙: 몸통>{rule['trunk_lean_deg']}° & 양 무릎>{rule['knee_straight_deg']}°)")

    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    t = df["timestamp_sec"].astype(float)
    ax[0].plot(t, df["trunk_lean_deg"].astype(float), label="trunk lean (deg)")
    ax[0].axhline(rule["trunk_lean_deg"], color="r", ls="--", label=f"{rule['trunk_lean_deg']}°")
    ax[0].legend(loc="upper right")
    ax[1].plot(t, df["knee_left_deg"].astype(float), label="knee L")
    ax[1].plot(t, df["knee_right_deg"].astype(float), label="knee R")
    ax[1].axhline(rule["knee_straight_deg"], color="r", ls="--", label=f"{rule['knee_straight_deg']}°")
    ax[1].legend(loc="lower right")
    ax[1].set_xlabel("sec")
    fig.suptitle(f"{args.video} ({clip['action_id']})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.video}_waist.png", dpi=120)
    print(f"[저장] {csv_path}\n[저장] {out_dir / f'{args.video}_waist.png'}")


def _names():
    from src.pose_utils import COCO_KEYPOINT_NAMES
    return COCO_KEYPOINT_NAMES


if __name__ == "__main__":
    main()
