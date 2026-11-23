"""8일차 — 다중 인물 Tracking: VICON 영상의 '뒤에 앉은 사람들'을 어떻게 처리하는지 확인.

실행:  uv run python scripts/d08_multi_person.py --video C012_T001
결과:  reports/day08/<video>_tracks.csv   Track 별 등장 Frame·누적 면적·판정 요약
       reports/day08/<video>_main_switch.csv  1초마다 '수행자(가장 큰 누적 면적)' ID 기록
확인:  ① 수행자 ID 가 영상 내내 바뀌지 않는지 (main_switch_count = 0 이 이상적)
       ② 뒤에 앉은 사람들의 판정은 무엇인지 → 서비스에서는 수행자만 저장할지, 모두 저장할지 결정
          (과제 T04 '두 사람 → Track ID별 결과 분리' 는 --all-tracks 로 보여 줄 수 있음)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd

from src.common import MODEL_DIR, REPORT_DIR, ensure_dir, load_clips, load_json, video_path
from src.pose_extract import load_pose_model
from src.realtime_engine import StreamAnalyzer, load_rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="C012_T001")
    ap.add_argument("--mode", default="GENERAL", choices=["GENERAL", "LIFT"])
    args = ap.parse_args()

    rf, _, info = load_rf(MODEL_DIR)
    pose, pcfg = load_pose_model()
    an = StreamAnalyzer(pose, pcfg, rf, info, load_json("waist_rule.json"), mode=args.mode, main_only=False)
    clip = next(c for c in load_clips() if c["video"] == args.video)
    cap = cv2.VideoCapture(str(video_path(clip)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_seen, main_log, n = {}, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        _, persons = an.feed(frame, t)
        for p in persons:
            frames_seen[p["track_id"]] = frames_seen.get(p["track_id"], 0) + 1
        if n % int(round(fps)) == 0:
            main_log.append({"t": round(t, 1), "main_track_id": an.main_track()})
        n += 1
    cap.release()

    out = ensure_dir(REPORT_DIR / "day08")
    rows = []
    for s in an.summary():
        tid = s["track_id"]
        rows.append({"track_id": tid, "is_main": s["is_main"], "frames": frames_seen.get(tid, 0),
                     "frame_ratio": round(frames_seen.get(tid, 0) / max(n, 1), 3),
                     "cum_box_area": int(an.area.get(tid, 0)), "result": s["result"], "reason": s["reason"],
                     "window_count": s["window_count"], "ng_window_count": s["ng_window_count"]})
    df = pd.DataFrame(rows).sort_values("cum_box_area", ascending=False)
    df.to_csv(out / f"{args.video}_tracks.csv", index=False, encoding="utf-8-sig")
    ml = pd.DataFrame(main_log)
    ml.to_csv(out / f"{args.video}_main_switch.csv", index=False, encoding="utf-8-sig")
    switches = int((ml["main_track_id"] != ml["main_track_id"].shift()).sum() - 1) if len(ml) else 0
    print(df.to_string(index=False))
    print(f"\n[Track 수] {len(df)}  [수행자 ID 변경 횟수] {switches}")
    print(f"[저장] {out / (args.video + '_tracks.csv')}")


if __name__ == "__main__":
    main()
