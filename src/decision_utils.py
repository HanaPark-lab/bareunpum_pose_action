"""바른품 통합 판정 (C안 §4.4) — 7·8일차 실시간과 내일 FastAPI 서비스가 같이 쓰는 코드.

창 1개 판정:  NG(위험 반복 검출) > CAUTION(허리, LIFT 모드에서만) > OK  /  정보 부족 = UNKNOWN
영상 1개 요약: 사람(Track)별로 가장 심각한 창 결과를 대표값으로.

'위험 없음'을 확인하지 못했는데 OK 로 떨어지는 else 는 두지 않는다:
  RF 를 돌릴 수 없으면(유효 Frame 부족) → UNKNOWN.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

DANGER = ("shake", "drop", "look_away")
SEVERITY = {"NG": 3, "CAUTION": 2, "OK": 1, "UNKNOWN": 0}

MESSAGES = {  # C안 §8 화면 문구
    ("OK", None): "현재 분석에서 주의가 필요한 동작이 감지되지 않았어요.",
    ("NG", "shake"): "강한 흔들림과 유사한 움직임이 감지됐어요. 동작을 확인해 주세요.",
    ("NG", "drop"): "낙상 관련 주의 움직임이 감지됐어요. 아기 위치를 확인해 주세요.",
    ("NG", "look_away"): "보호자의 주의 이탈과 유사한 움직임이 감지됐어요. 주변을 확인해 주세요.",
    ("CAUTION", "waist"): "허리에 부담이 갈 수 있어요. 무릎을 굽히고 몸에 가깝게 들어 올려 주세요.",
    ("UNKNOWN", "low_quality"): "자세를 정확히 확인하기 어려워요. 촬영 위치를 조정해 주세요.",
}


def message(result: str, reason: str | None) -> str:
    return MESSAGES.get((result, reason), "")


class TrackDecider:
    """한 사람(Track)의 연속 창 판정. 같은 위험이 repeat_windows 번 연속이면 NG."""

    def __init__(self, classes, danger_threshold=0.70, repeat_windows=2, mode="GENERAL"):
        self.classes = list(classes)
        self.thr = danger_threshold
        self.repeat = repeat_windows
        self.mode = mode
        self._streak_label, self._streak = None, 0

    def decide(self, proba: np.ndarray | None, waist: dict | None) -> dict:
        """proba: RF predict_proba 1줄 (None = 창 부족). waist: waist_utils.window_waist_result"""
        out = {"result": "UNKNOWN", "reason": "low_quality", "action_label": None, "confidence": None,
               "danger_label": None, "danger_prob": None,
               "waist_rule_applied": False, "trunk_lean_deg": None}
        if waist is not None:
            out["trunk_lean_deg"] = waist.get("max_trunk_lean_deg")
        if proba is None:
            self._streak_label, self._streak = None, 0
            return out
        top = int(np.argmax(proba))
        out["action_label"] = self.classes[top]
        out["confidence"] = round(float(proba[top]), 3)
        dprobs = {c: float(proba[self.classes.index(c)]) for c in DANGER if c in self.classes}
        best = max(dprobs, key=dprobs.get) if dprobs else None
        if best is not None:
            out["danger_label"], out["danger_prob"] = best, round(dprobs[best], 3)
        # ① 위험 반복 검출
        if best is not None and dprobs[best] >= self.thr:
            self._streak = self._streak + 1 if self._streak_label == best else 1
            self._streak_label = best
        else:
            self._streak_label, self._streak = None, 0
        if self._streak >= self.repeat:
            out.update(result="NG", reason=self._streak_label)
            return out
        # ② 허리 규칙 (들어 올리기 모드에서만)
        if self.mode == "LIFT" and waist is not None and waist.get("applied"):
            out["waist_rule_applied"] = True
            if waist.get("violation"):
                out.update(result="CAUTION", reason="waist")
                return out
        # ③ 위험 미검출 + (적용 가능한) 허리 검사 통과
        out.update(result="OK", reason=None)
        return out


def summarize_track(window_results: list[dict]) -> dict:
    """창 결과 목록 → 사람 1명 대표 결과 (FastAPI 응답의 predictions 한 항목)"""
    if not window_results:
        return {"result": "UNKNOWN", "reason": "low_quality", "window_count": 0}
    worst = max(window_results, key=lambda r: SEVERITY[r["result"]])
    ng = [r for r in window_results if r["result"] == "NG"]
    reason = Counter(r["reason"] for r in ng).most_common(1)[0][0] if ng else worst["reason"]
    labels = [r["action_label"] for r in window_results if r["action_label"]]
    confs = [r["confidence"] for r in window_results if r["confidence"] is not None]
    leans = [r["trunk_lean_deg"] for r in window_results if r["trunk_lean_deg"] is not None]
    return {
        "result": worst["result"],
        "reason": reason,
        "message": message(worst["result"], reason),
        # NG 이면 대표 동작 = 위험 사유, 아니면 가장 많이 나온 RF 예측
        "action_label": reason if ng else (Counter(labels).most_common(1)[0][0] if labels else None),
        "confidence": round(float(np.mean(confs)), 3) if confs else None,
        "window_count": len(window_results),
        "ng_window_count": len(ng),
        "unknown_window_count": sum(r["result"] == "UNKNOWN" for r in window_results),
        "waist_rule_applied": any(r["waist_rule_applied"] for r in window_results),
        "trunk_lean_deg": round(max(leans), 1) if leans else None,
    }
