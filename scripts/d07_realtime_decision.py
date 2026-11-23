"""7일차 — 저장 영상 또는 IP Webcam 을 실시간처럼 흘려 1초마다 OK/주의/NG/UNKNOWN 판정.

실행 (저장 영상):  uv run python scripts/d07_realtime_decision.py --video C003_T001
     (허리 모드):  uv run python scripts/d07_realtime_decision.py --video C004_T001 --mode LIFT
     (IP Webcam):  uv run python scripts/d07_realtime_decision.py --url http://192.168.0.10:8080/video --seconds 8
     (8일차 다중):  --all-tracks  → 뒤에 앉은 사람까지 모두 판정 (기본은 수행자 1명)
결과:  reports/day07/<이름>_decision.mp4   화면에 결과 색상·사유·몸통 기울기 표시
       reports/day07/<이름>_windows.csv    1초마다의 창 판정 기록
       화면 출력: 사람별 요약 (내일 FastAPI 응답의 predictions 와 같은 모양)
확인:  흔들기 구간에서만 NG 가 뜨는지, 안아 올리기·내려놓기에서 오경보가 나는지,
       --mode LIFT 에서 무릎 펴고 숙일 때만 CAUTION 이 뜨는지.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd

from src.common import MODEL_DIR, REPORT_DIR, ensure_dir, load_clips, load_json, video_path
from src.pose_extract import load_pose_model
from src.realtime_engine import StreamAnalyzer, load_rf

COLORS = {"OK": (80, 180, 80), "CAUTION": (0, 200, 255), "NG": (40, 40, 230), "UNKNOWN": (160, 160, 160)}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="clips.csv 의 video 이름")
    g.add_argument("--url", help="IP Webcam 주소 (예: http://IP:8080/video)")
    ap.add_argument("--mode", default="GENERAL", choices=["GENERAL", "LIFT"])
    ap.add_argument("--seconds", type=float, default=None, help="이 시간만 분석 (IP Webcam 은 8 권장)")
    ap.add_argument("--all-tracks", action="store_true")
    args = ap.parse_args()

    rf, _, info = load_rf(MODEL_DIR)
    print(f"[모델] {info['model_version']}  classes={info['classes']}  features={info['feature_count']}")
    pose, pcfg = load_pose_model()
    analyzer = StreamAnalyzer(pose, pcfg, rf, info, load_json("waist_rule.json"), mode=args.mode,
                              main_only=not args.all_tracks)

    if args.video:
        clip = next(c for c in load_clips() if c["video"] == args.video)
        src, name, live = str(video_path(clip)), args.video, False
    else:
        src, name, live = args.url, "ipwebcam_" + time.strftime("%H%M%S"), True
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"입력을 열 수 없습니다: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_dir = ensure_dir(REPORT_DIR / "day07")
    writer, n, t0 = None, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (time.time() - t0) if live else n / fps
        if args.seconds and t > args.seconds:
            break
        _, persons = analyzer.feed(frame, t)
        main = analyzer.main_track()
        for p in persons:
            res = analyzer.latest.get(p["track_id"])
            if res is None or (not args.all_tracks and p["track_id"] != main):
                continue
            x1, y1, x2, y2 = map(int, p["box"])
            c = COLORS[res["result"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
            text = f"ID{p['track_id']} {res['result']} {res['reason'] or ''}"
            if res["trunk_lean_deg"] is not None:
                text += f" lean {res['trunk_lean_deg']:.0f}"
            cv2.putText(frame, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        cv2.putText(frame, f"{t:5.1f}s  mode={args.mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_dir / f"{name}_decision.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        n += 1
    cap.release()
    if writer:
        writer.release()

    rows = [dict(track_id=tid, **r) for tid, rs in analyzer.results.items() for r in rs]
    pd.DataFrame(rows).to_csv(out_dir / f"{name}_windows.csv", index=False, encoding="utf-8-sig")
    summary = analyzer.summary()
    print(f"\n[처리] {n} frames, {time.time() - t0:.1f}s")
    print(json.dumps({"model_version": info["model_version"], "predictions": summary}, ensure_ascii=False, indent=2, default=str))
    print(f"\n[저장] {out_dir / (name + '_decision.mp4')}, {out_dir / (name + '_windows.csv')}")


if __name__ == "__main__":
    main()
