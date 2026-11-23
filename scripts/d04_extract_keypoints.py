"""4·5일차 ① — 모든 영상에서 수행자 1명의 Keypoint CSV 추출 (수업 26_extract_keypoint_dataset 대응).

실행:  uv run python scripts/d04_extract_keypoints.py            # clips.csv 전체 (이미 있으면 건너뜀)
       uv run python scripts/d04_extract_keypoints.py --force    # 다시 추출
       uv run python scripts/d04_extract_keypoints.py --external # HMDB51 외부 데이터까지
결과:  data/keypoints/<video>.csv   (frame_id, timestamp_sec, track_id, pose_detected, {관절}_x_px/_y_px/_conf)
       reports/day04/extract_summary.csv
시간:  CPU 기준 영상 1초당 약 1.5초 (VICON 20개 ≈ 15~20분). GPU면 훨씬 빠름.
확인:  extract_summary 의 detect_ratio 가 0.9 이상인지, track_count 가 큰 VICON 영상에서
       main_track_frames 가 영상 Frame 수와 비슷한지(수행자가 끝까지 같은 ID인지).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import KEYPOINT_DIR, REPORT_DIR, ensure_dir, load_clips, load_external_clips, video_path
from src.pose_extract import keypoint_columns, load_pose_model, main_person_rows, select_main_track, track_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--external", action="store_true", help="clips_external.csv (HMDB51) 도 추출")
    args = ap.parse_args()

    ensure_dir(KEYPOINT_DIR)
    out_dir = ensure_dir(REPORT_DIR / "day04")
    model, config = load_pose_model()
    summary_rows = []
    clips = load_clips() + (load_external_clips() if args.external else [])
    for clip in clips:
        if args.only and clip["video"] not in args.only:
            continue
        if clip["source"] == "vicon2d":
            continue  # 영상이 아니라 이미 Keypoint CSV
        if clip["action_id"] == "":
            print(f"[건너뜀] {clip['video']}: clips.csv 에 action_id 가 비어 있음")
            continue
        path = video_path(clip)
        out_csv = KEYPOINT_DIR / f"{clip['video']}.csv"
        if not path.exists():
            print(f"[없음] {path}")
            continue
        if out_csv.exists() and not args.force:
            print(f"[있음] {out_csv.name}")
            continue
        frames, fps = track_video(path, model, config)
        main_id, summary = select_main_track(frames)
        rows = main_person_rows(frames, main_id)
        fields = ["frame_id", "timestamp_sec", "track_id", "pose_detected", "person_count", "box_conf"] + keypoint_columns()
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        detected = sum(r["pose_detected"] for r in rows)
        s = {"video": clip["video"], "source": clip["source"], "action_id": clip["action_id"],
             "fps": round(fps, 2), "frames": len(rows), "track_count": summary.get("track_count", 0),
             "main_track_id": main_id, "main_track_frames": summary.get("main_track_frames", 0),
             "detect_ratio": round(detected / max(len(rows), 1), 3)}
        summary_rows.append(s)
        print(f"[저장] {out_csv.name:36s} 사람 {s['track_count']:2d}명  수행자 Frame {s['main_track_frames']}/{s['frames']}")

    if summary_rows:
        p = out_dir / "extract_summary.csv"
        with p.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n[저장] {p}")


if __name__ == "__main__":
    main()
