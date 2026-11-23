"""영상 → YOLO Pose + ByteTrack → Frame별 사람 목록, 그리고 '수행자 1명' 고르기.

VICON 영상은 수행자 뒤에 4~6명이 앉아 있으므로, 수업 5일차처럼 '매 Frame 가장 큰 박스'를
고르면 Frame마다 다른 사람이 섞일 수 있다. 그래서 8일차 Tracking으로 Track ID를 붙이고,
영상 전체에서 '박스 면적 합'이 가장 큰 Track 1개를 수행자로 정한다.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.common import load_config, resolve_device
from src.pose_utils import COCO_KEYPOINT_NAMES


def load_pose_model():
    config = load_config()
    return YOLO(config["pose_model"]), config


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def track_video(path: Path, model=None, config=None, frame_step: int = 1, max_seconds: float | None = None):
    """영상 전체를 Tracking하며 Frame별 사람 목록을 돌려준다.

    반환: frames = [{frame_id, timestamp_sec, persons:[{track_id, box(x1,y1,x2,y2), box_conf, kp(17,2), kp_conf(17)}]}]
    """
    if model is None:
        model, config = load_pose_model()
    device = resolve_device(config["device"])
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, frame_id = [], -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if max_seconds is not None and frame_id / fps > max_seconds:
            break
        if frame_id % frame_step:
            continue
        result = model.track(
            frame, persist=True, tracker=config.get("tracker", "bytetrack.yaml"),
            imgsz=config["image_size"], conf=config["person_confidence"],
            device=device, verbose=False,
        )[0]
        persons = []
        if result.boxes is not None and result.keypoints is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            bconf = result.boxes.conf.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else np.full(len(boxes), -1)
            kxy = result.keypoints.xy.cpu().numpy()
            kcf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.ones(kxy.shape[:2])
            for i in range(len(boxes)):
                persons.append({"track_id": int(ids[i]), "box": boxes[i], "box_conf": float(bconf[i]),
                                "kp": kxy[i], "kp_conf": kcf[i]})
        frames.append({"frame_id": frame_id, "timestamp_sec": frame_id / fps, "persons": persons})
    cap.release()
    # 다음 영상에서 Track ID가 이어지지 않도록 tracker 초기화
    if getattr(model, "predictor", None) is not None and getattr(model.predictor, "trackers", None):
        for t in model.predictor.trackers:
            t.reset()
    return frames, fps


def box_area(box) -> float:
    return float(max(box[2] - box[0], 0) * max(box[3] - box[1], 0))


def select_main_track(frames) -> tuple[int | None, dict]:
    """영상 전체에서 박스 면적 합이 가장 큰 Track = 수행자."""
    score, seen = defaultdict(float), defaultdict(int)
    for fr in frames:
        for p in fr["persons"]:
            if p["track_id"] < 0:
                continue
            score[p["track_id"]] += box_area(p["box"])
            seen[p["track_id"]] += 1
    if not score:
        return None, {}
    main = max(score, key=score.get)
    summary = {"main_track_id": main, "main_track_frames": seen[main],
               "track_count": len(score), "frames": len(frames)}
    return main, summary


def main_person_rows(frames, main_track: int | None) -> list[dict]:
    """수업 5일차 Keypoint CSV와 같은 열 이름({관절}_x_px, _y_px, _conf, pose_detected)으로 변환."""
    rows = []
    for fr in frames:
        row = {"frame_id": fr["frame_id"], "timestamp_sec": round(fr["timestamp_sec"], 4),
               "track_id": "", "pose_detected": 0, "person_count": len(fr["persons"])}
        person = next((p for p in fr["persons"] if p["track_id"] == main_track), None)
        for j, name in enumerate(COCO_KEYPOINT_NAMES):
            row[f"{name}_x_px"] = round(float(person["kp"][j][0]), 2) if person is not None else ""
            row[f"{name}_y_px"] = round(float(person["kp"][j][1]), 2) if person is not None else ""
            row[f"{name}_conf"] = round(float(person["kp_conf"][j]), 4) if person is not None else ""
        if person is not None:
            row["track_id"] = main_track
            row["pose_detected"] = 1
            row["box_conf"] = round(person["box_conf"], 4)
        rows.append(row)
    return rows


def keypoint_columns() -> list[str]:
    cols = []
    for name in COCO_KEYPOINT_NAMES:
        cols += [f"{name}_x_px", f"{name}_y_px", f"{name}_conf"]
    return cols
