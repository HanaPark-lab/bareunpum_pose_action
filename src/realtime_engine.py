"""7·8일차 — Frame 스트림(저장 영상·IP Webcam)을 받아 사람별로 1초마다 판정.

흐름: Frame → YOLO Pose + ByteTrack → Track별 버퍼(시간, 38-D, 관절) → 최근 2초 창
      → RF predict_proba + 허리 규칙 → TrackDecider → OK / CAUTION / NG / UNKNOWN
내일 FastAPI 의 pose_pipeline·action_inference 가 이 파일을 그대로 가져다 쓴다.
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common import resolve_device
from src.decision_utils import TrackDecider, summarize_track
from src.pose_extract import box_area
from src.pose_utils import COCO_KEYPOINT_NAMES
from src.sequence_utils import frame_feature_from_row
from src.waist_utils import frame_waist_values, window_waist_result
from src.window_utils import window_vector


def load_rf(model_dir: Path):
    """모델 3개 로드 + 형식 확인 (과제 T01)"""
    model = joblib.load(model_dir / "rf_action_baseline.joblib")
    columns = json.loads((model_dir / "rf_feature_columns.json").read_text(encoding="utf-8"))
    info = json.loads((model_dir / "rf_training_info.json").read_text(encoding="utf-8"))
    assert model.n_features_in_ == len(columns) == info["feature_count"], "Feature 수 불일치"
    assert [str(c) for c in model.classes_] == info["classes"], "Class 순서 불일치"
    return model, columns, info


def _row_from_person(kp, kp_conf) -> pd.Series:
    row = {"pose_detected": 1}
    for j, name in enumerate(COCO_KEYPOINT_NAMES):
        row[f"{name}_x_px"], row[f"{name}_y_px"] = float(kp[j][0]), float(kp[j][1])
    return pd.Series(row)


class StreamAnalyzer:
    def __init__(self, pose_model, pose_config, rf, info, waist_rule, mode="GENERAL",
                 main_only=True, prediction_interval=1.0):
        self.pose, self.pcfg = pose_model, pose_config
        self.rf, self.info, self.rule = rf, info, waist_rule
        self.mode, self.main_only = mode, main_only
        self.interval = prediction_interval
        self.win = float(info["window_seconds"])
        self.buffers = defaultdict(lambda: deque())       # track → [(t, feat|None, waist_values)]
        self.deciders = {}
        self.results = defaultdict(list)                  # track → 창 결과 목록
        self.latest = {}                                  # track → 마지막 창 결과 (화면 표시용)
        self.area = defaultdict(float)                    # 누적 박스 면적 → 수행자 고르기
        self._last_pred_t = -1e9

    def main_track(self):
        return max(self.area, key=self.area.get) if self.area else None

    def feed(self, frame, t: float):
        r = self.pose.track(frame, persist=True, tracker=self.pcfg.get("tracker", "bytetrack.yaml"),
                            imgsz=self.pcfg["image_size"], conf=self.pcfg["person_confidence"],
                            device=resolve_device(self.pcfg["device"]), verbose=False)[0]
        persons = []
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            boxes = r.boxes.xyxy.cpu().numpy()
            kxy = r.keypoints.xy.cpu().numpy()
            kcf = r.keypoints.conf.cpu().numpy()
            for i, tid in enumerate(ids):
                self.area[tid] += box_area(boxes[i])
                feat = frame_feature_from_row(_row_from_person(kxy[i], kcf[i]))
                wv = frame_waist_values(kxy[i], kcf[i], self.rule["min_joint_conf"])
                self.buffers[tid].append((t, feat, wv))
                persons.append({"track_id": int(tid), "box": boxes[i]})
        for buf in self.buffers.values():                 # 오래된 Frame 버리기
            while buf and buf[0][0] < t - self.win:
                buf.popleft()
        if t - self._last_pred_t >= self.interval:
            self._last_pred_t = t
            targets = [self.main_track()] if self.main_only else list(self.buffers)
            for tid in targets:
                if tid is not None and self.buffers[tid]:   # 사라진 사람은 판정하지 않음
                    self._predict(tid, t)
        return r, persons

    def _predict(self, tid, t):
        buf = list(self.buffers[tid])
        valid = [(bt, f) for bt, f, _ in buf if f is not None]
        proba = None
        if valid:
            times = np.array([v[0] for v in valid])
            feats = np.stack([v[1] for v in valid])
            vec = window_vector(times, feats, t - self.win, t + 1e-6,
                                self.info["sequence_steps"], self.info["min_valid_frames"])
            if vec is not None:
                proba = self.rf.predict_proba(vec.reshape(1, -1))[0]
        waist = window_waist_result([w for _, _, w in buf], self.rule) if buf else None
        if tid not in self.deciders:
            self.deciders[tid] = TrackDecider(self.info["classes"], self.info["danger_threshold"],
                                              self.info["repeat_windows"], self.mode)
        res = self.deciders[tid].decide(proba, waist)
        res["t"] = round(t, 2)
        self.results[tid].append(res)
        self.latest[tid] = res

    def summary(self) -> list[dict]:
        main = self.main_track()
        out = []
        for tid, rs in self.results.items():
            s = summarize_track(rs)
            s["track_id"] = int(tid)
            s["is_main"] = bool(tid == main)
            out.append(s)
        return sorted(out, key=lambda s: (not s["is_main"], s["track_id"]))
