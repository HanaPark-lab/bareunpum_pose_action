"""바른품 허리 규칙 (C안 §4.3).

- trunk_lean_2d_deg: 골반 중점 → 어깨 중점 벡터와 화면 위쪽(0,-1)의 각도. 0° = 똑바로 섬.
- 화면상 2D 기울기일 뿐 임상적 허리 굴곡각이 아님.
- 45° / 160° / 30% 는 검증 전 팀 가정값 (configs/waist_rule.json).
"""
from __future__ import annotations

import numpy as np

from src.feature_utils import calculate_angle

L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 5, 6, 11, 12, 13, 14, 15, 16
TRUNK_JOINTS = [L_SH, R_SH, L_HIP, R_HIP]
KNEE_JOINTS = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK]


def trunk_lean_2d_deg(kp: np.ndarray) -> float:
    """kp: (17, 2) 픽셀 좌표"""
    hip_mid = (kp[L_HIP] + kp[R_HIP]) / 2.0
    shoulder_mid = (kp[L_SH] + kp[R_SH]) / 2.0
    v = shoulder_mid - hip_mid
    up = np.array([0.0, -1.0])
    cos = np.dot(v, up) / (np.linalg.norm(v) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def frame_waist_values(kp: np.ndarray, conf: np.ndarray, min_conf: float = 0.5) -> dict:
    """한 Frame의 몸통 기울기·양 무릎 각도. 필요한 관절이 conf 미달이면 None."""
    out = {"trunk_lean_deg": None, "knee_left_deg": None, "knee_right_deg": None}
    if np.all(conf[TRUNK_JOINTS] >= min_conf):
        out["trunk_lean_deg"] = trunk_lean_2d_deg(kp)
    if np.all(conf[[L_HIP, L_KNEE, L_ANK]] >= min_conf):
        out["knee_left_deg"] = calculate_angle(kp[L_HIP], kp[L_KNEE], kp[L_ANK])
    if np.all(conf[[R_HIP, R_KNEE, R_ANK]] >= min_conf):
        out["knee_right_deg"] = calculate_angle(kp[R_HIP], kp[R_KNEE], kp[R_ANK])
    return out


def frame_violates(v: dict, rule: dict) -> bool | None:
    """허리를 많이 숙이면서(>45°) 무릎을 거의 펴고(>160°) 있으면 위반.

    한쪽 무릎 꿇기 오경보를 줄이려고 '양쪽 무릎이 모두 펴짐'일 때만 위반으로 본다.
    판단에 필요한 값이 없으면 None (UNKNOWN 후보).
    """
    if v["trunk_lean_deg"] is None or v["knee_left_deg"] is None or v["knee_right_deg"] is None:
        return None
    bent_back = v["trunk_lean_deg"] > rule["trunk_lean_deg"]
    straight_knees = min(v["knee_left_deg"], v["knee_right_deg"]) > rule["knee_straight_deg"]
    return bool(bent_back and straight_knees)


def window_waist_result(frame_values: list[dict], rule: dict) -> dict:
    """창(window) 안의 Frame들로 허리 규칙 판정.

    반환: applied(판정 가능 여부), violation(bool|None), violation_ratio, max_trunk_lean_deg, valid_ratio
    """
    flags = [frame_violates(v, rule) for v in frame_values]
    valid = [f for f in flags if f is not None]
    leans = [v["trunk_lean_deg"] for v in frame_values if v["trunk_lean_deg"] is not None]
    valid_ratio = len(valid) / max(len(flags), 1)
    if valid_ratio < 0.5:
        return {"applied": False, "violation": None, "violation_ratio": None,
                "max_trunk_lean_deg": max(leans) if leans else None, "valid_ratio": valid_ratio}
    ratio = sum(valid) / len(valid)
    return {"applied": True, "violation": ratio >= rule["violation_frame_ratio"],
            "violation_ratio": ratio, "max_trunk_lean_deg": max(leans) if leans else None,
            "valid_ratio": valid_ratio}
