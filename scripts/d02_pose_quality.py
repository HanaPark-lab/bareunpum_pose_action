"""2일차 — Keypoint Confidence·Pose 품질 확인 (모든 영상, 수행자 기준).

실행:  uv run python scripts/d02_pose_quality.py            # 모든 영상 (5 frame마다)
       uv run python scripts/d02_pose_quality.py --only C003_T001 KakaoTalk_20261123_151939449
결과:  reports/day02/pose_quality.csv  (영상 1줄)
       reports/day02/joint_conf.csv    (영상 × 관절 평균 conf)
확인:  detect_ratio(수행자가 잡힌 비율), track_count(사람 수), 하체 관절 conf.
       허리 규칙은 골반·무릎·발목 conf ≥ 0.5 가 필요하다 → lower_body_valid_ratio 가 낮으면 허리 판정은 UNKNOWN.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.common import REPORT_DIR, ensure_dir, load_clips, video_path
from src.pose_extract import load_pose_model, select_main_track, track_video, video_info
from src.pose_utils import COCO_KEYPOINT_NAMES

UPPER = [5, 6, 7, 8, 9, 10]          # 어깨·팔꿈치·손목
LOWER = [11, 12, 13, 14, 15, 16]     # 골반·무릎·발목


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--frame-step", type=int, default=5, help="속도를 위해 n frame마다 1장")
    args = ap.parse_args()

    out_dir = ensure_dir(REPORT_DIR / "day02")
    model, config = load_pose_model()
    clips = [c for c in load_clips() if args.only is None or c["video"] in args.only]

    q_rows, j_rows = [], []
    for clip in clips:
        path = video_path(clip)
        if not path.exists():
            print(f"[없음] {path}")
            continue
        info = video_info(path)
        frames, fps = track_video(path, model, config, frame_step=args.frame_step)
        main_id, summary = select_main_track(frames)
        confs = []
        for fr in frames:
            p = next((p for p in fr["persons"] if p["track_id"] == main_id), None)
            if p is not None:
                confs.append(p["kp_conf"])
        confs = np.array(confs) if confs else np.zeros((0, 17))
        detect_ratio = len(confs) / max(len(frames), 1)
        lower_ok = float(np.mean(np.all(confs[:, LOWER] >= 0.5, axis=1))) if len(confs) else 0.0
        row = {
            "video": clip["video"], "source": clip["source"], "action_id": clip["action_id"],
            "width": info["width"], "height": info["height"], "fps": round(info["fps"], 2),
            "seconds": round(info["frames"] / info["fps"], 1),
            "track_count": summary.get("track_count", 0), "main_track_id": main_id,
            "detect_ratio": round(detect_ratio, 3),
            "upper_conf_mean": round(float(confs[:, UPPER].mean()), 3) if len(confs) else "",
            "lower_conf_mean": round(float(confs[:, LOWER].mean()), 3) if len(confs) else "",
            "lower_body_valid_ratio": round(lower_ok, 3),
        }
        q_rows.append(row)
        jr = {"video": clip["video"]}
        for j, name in enumerate(COCO_KEYPOINT_NAMES):
            jr[name] = round(float(confs[:, j].mean()), 3) if len(confs) else ""
        j_rows.append(jr)
        print(f"{clip['video']:32s} 사람 {row['track_count']:2d}명  수행자검출 {row['detect_ratio']:.2f}  "
              f"상체 {row['upper_conf_mean']}  하체 {row['lower_conf_mean']}  하체유효 {row['lower_body_valid_ratio']}")

    for name, rows in [("pose_quality.csv", q_rows), ("joint_conf.csv", j_rows)]:
        if rows:
            with (out_dir / name).open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"\n[저장] {out_dir / 'pose_quality.csv'}, {out_dir / 'joint_conf.csv'}")


if __name__ == "__main__":
    main()
