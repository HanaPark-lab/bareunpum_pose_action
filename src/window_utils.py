"""시간 창(window) 만들기 — 학습(5·6일차)과 실시간(7일차)·서비스가 똑같이 쓰는 코드.

수업 6일차는 '클립 1개 = 샘플 1개'(클립 전체를 30 Step으로 리샘플)였다.
바른품 영상은 한 영상 안에 안아 올리기 → 핵심 동작 → 내려놓기가 이어지므로,
2초 창을 0.5초씩 밀며 잘라 '창 1개 = 샘플 1개'로 만든다. 창 하나를 30 Step으로 리샘플하므로
Feature 형식(30 × 38 = 1,140)은 수업과 똑같다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sequence_utils import (
    FRAME_FEATURE_NAMES,
    flatten_sequence,
    frame_feature_from_row,
    resample_sequence,
)


def frame_feature_table(kp_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Keypoint CSV → (유효 Frame의 timestamp 배열, (T, 38) Feature 배열)"""
    times, feats = [], []
    for _, row in kp_df.sort_values("frame_id").iterrows():
        if int(row.get("pose_detected", 0) or 0) != 1:
            continue
        f = frame_feature_from_row(row)
        if f is None:
            continue
        times.append(float(row["timestamp_sec"]))
        feats.append(f)
    if not feats:
        return np.zeros(0), np.zeros((0, len(FRAME_FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(times), np.stack(feats)


def window_vector(times: np.ndarray, feats: np.ndarray, start: float, end: float,
                  steps: int, min_valid_frames: int):
    """[start, end) 구간의 Frame Feature를 steps 개로 리샘플해 1차원 벡터로. 부족하면 None."""
    mask = (times >= start) & (times < end)
    if mask.sum() < max(min_valid_frames, 2):
        return None
    seq = resample_sequence(feats[mask], times[mask], target_steps=steps)
    return flatten_sequence(seq)


def sliding_windows(times: np.ndarray, window_sec: float, stride_sec: float):
    """영상 시간 범위에서 (start, end) 창 목록"""
    if len(times) == 0:
        return []
    t0, t1 = float(times.min()), float(times.max())
    out, s = [], t0
    while s + window_sec <= t1 + 1e-6:
        out.append((s, s + window_sec))
        s += stride_sec
    return out


def window_label(center: float, base_label: str, seg_start, seg_end, margin: float) -> str | None:
    """창 중심 시각으로 라벨 결정.

    - 구간(start~end)이 없으면 영상 전체가 base_label
    - 구간 안쪽(경계에서 margin 이상 안쪽) → base_label
    - 구간 바깥(경계에서 margin 이상 바깥) → OTHER (안아 올리기·내려놓기 같은 일반 돌봄)
    - 경계 근처 → None (애매하므로 학습에서 제외)
    """
    if seg_start is None or seg_end is None:
        return base_label
    if seg_start + margin <= center <= seg_end - margin:
        return base_label
    if center < seg_start - margin or center > seg_end + margin:
        return "OTHER"
    return None


def build_clip_windows(kp_df: pd.DataFrame, base_label: str, seg_start, seg_end, cfg: dict):
    """Keypoint CSV 1개 → [(vector, label, start, end)] (d05 학습 Dataset, d09 보강이 같이 사용)"""
    times, feats = frame_feature_table(kp_df)
    out = []
    for s, e in sliding_windows(times, cfg["window_seconds"], cfg["window_stride_seconds"]):
        label = window_label((s + e) / 2, base_label, seg_start, seg_end, cfg["boundary_margin_seconds"])
        if label is None:
            continue
        vec = window_vector(times, feats, s, e, cfg["sequence_steps"], cfg["min_valid_frames"])
        if vec is not None:
            out.append((vec, label, s, e))
    return out
