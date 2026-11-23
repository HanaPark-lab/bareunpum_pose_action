"""1일차 — 환경 점검 + 바른품 영상 1개에 YOLO Pose 적용해 Skeleton 영상 만들기.

실행:  uv run python scripts/d01_env_and_pose_video.py --video C003_T001 --seconds 15
결과:  reports/day01/C003_T001_pose.mp4, reports/day01/env.txt
확인:  수행자(검은 슈트)와 뒤에 앉은 사람들이 모두 잡히는지, 수행자 관절이 화면에 붙어 있는지
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import torch
import ultralytics

from src.common import REPORT_DIR, ensure_dir, load_clips, load_config, resolve_device, video_path
from src.pose_extract import load_pose_model, video_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="C003_T001", help="clips.csv 의 video 이름")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    out_dir = ensure_dir(REPORT_DIR / "day01")
    config = load_config()

    # ① 실행 환경 기록
    env = [
        f"python      : {platform.python_version()}",
        f"opencv      : {cv2.__version__}",
        f"torch       : {torch.__version__}  cuda={torch.cuda.is_available()}",
        f"ultralytics : {ultralytics.__version__}",
        f"device      : {resolve_device(config['device'])}",
        f"pose_model  : {config['pose_model']}",
    ]
    (out_dir / "env.txt").write_text("\n".join(env) + "\n", encoding="utf-8")
    print("\n".join(env))

    # ② 영상 찾기
    clip = next((c for c in load_clips() if c["video"] == args.video), None)
    if clip is None:
        raise SystemExit(f"clips.csv 에 {args.video} 가 없습니다.")
    path = video_path(clip)
    info = video_info(path)
    print(f"\n[영상] {path.name}  {info['width']}x{info['height']}  {info['fps']:.1f}fps  {info['frames']} frames")

    # ③ Pose 추론 → Skeleton 그린 영상 저장 (모든 사람)
    model, _ = load_pose_model()
    cap = cv2.VideoCapture(str(path))
    out_path = out_dir / f"{args.video}_pose.mp4"
    writer = None
    n, persons_total = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or n / info["fps"] > args.seconds:
            break
        result = model.predict(frame, imgsz=config["image_size"], conf=config["person_confidence"],
                               device=resolve_device(config["device"]), verbose=False)[0]
        drawn = result.plot()
        persons_total += 0 if result.boxes is None else len(result.boxes)
        if writer is None:
            h, w = drawn.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (w, h))
        writer.write(drawn)
        n += 1
    cap.release()
    if writer:
        writer.release()
    print(f"[저장] {out_path}  ({n} frames, Frame당 평균 사람 수 {persons_total / max(n, 1):.1f})")


if __name__ == "__main__":
    main()
