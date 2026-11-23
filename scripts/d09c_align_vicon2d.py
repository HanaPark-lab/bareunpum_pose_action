"""9일차 확장 — VICON 2D(d09b 결과)와 RGB 영상의 시간차를 자동으로 맞추고 학습 목록 만들기.

왜 필요?  FBX 와 휴대폰 영상은 녹화 시작 시각이 달라서, clips.csv 에 '영상 기준'으로 적은
         위험 구간(예: 흔들기 4~13초)을 FBX 시간으로 옮기려면 시간차(offset)가 필요하다.
방법:    같은 Take 의 RGB Keypoint CSV(d04)와 VICON 2D 16개 시점 각각에서
         '골반 기준 손목·코·무릎 높이 / 몸통 길이' 신호를 만들고, ±6초 안에서 상관이 가장 큰 시간차를 찾는다.
         가장 잘 맞는 시점(yaw)이 실제 카메라 방향과 비슷한지도 같이 출력된다.

실행:  uv run python scripts/d09c_align_vicon2d.py
결과:  data/labels/clips_vicon2d.csv  (source=vicon2d, P001, Train 전용, 구간은 FBX 시간으로 변환)
       reports/day09/vicon2d_alignment.csv  (take, best_view, offset_sec, corr)
옵션:  --include-other  OTHER Take 도 16시점 모두 넣기 (기본은 위험 Take 만).
       OTHER 를 다 넣으면 OTHER 창이 1만 개 넘게 늘어 위험 확률이 다시 묻힌다 (11/23 실험: E2 shake recall 0).
확인:  corr 이 0.6 미만인 Take 는 시간차가 불확실 → 위험 Take 면 학습에서 빠진다(OTHER Take 는 영상 전체라 괜찮음).
이후:  d05 → d06 (E2 = E1 + VICON 2D 가 자동으로 비교됨)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import DATA_DIR, KEYPOINT_DIR, REPORT_DIR, ensure_dir, load_actions, load_clips

VICON2D = DATA_DIR / "vicon2d"
FPS = 30.0
MAX_LAG = 6.0
MIN_CORR = 0.6


def height_signal(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """시간(30fps 격자)과 (T,4) 신호: 왼손목·오른손목·코·무릎평균 의 높이(골반 기준, 몸통 길이로 나눔)"""
    d = df[df["pose_detected"] == 1].copy()
    hip_y = (d["left_hip_y_px"] + d["right_hip_y_px"]) / 2
    sh_y = (d["left_shoulder_y_px"] + d["right_shoulder_y_px"]) / 2
    torso = (hip_y - sh_y).abs().clip(lower=1.0)
    cols = [d["left_wrist_y_px"], d["right_wrist_y_px"], d["nose_y_px"],
            (d["left_knee_y_px"] + d["right_knee_y_px"]) / 2]
    sig = np.stack([((hip_y - c) / torso).to_numpy() for c in cols], axis=1)
    t = d["timestamp_sec"].to_numpy(dtype=float)
    grid = np.arange(t.min(), t.max(), 1 / FPS)
    out = np.stack([np.interp(grid, t, sig[:, i]) for i in range(sig.shape[1])], axis=1)
    out = (out - out.mean(0)) / (out.std(0) + 1e-6)
    return grid, out


def best_lag(video_sig, vicon_sig):
    """vicon 시간 = video 시간 - lag 가 되도록 하는 lag(초)와 상관계수"""
    best = (0.0, -1.0)
    n = int(MAX_LAG * FPS)
    for k in range(-n, n + 1):
        if k >= 0:
            a, b = video_sig[k:], vicon_sig
        else:
            a, b = video_sig, vicon_sig[-k:]
        m = min(len(a), len(b))
        if m < 3 * FPS:
            continue
        c = float(np.mean([np.corrcoef(a[:m, i], b[:m, i])[0, 1] for i in range(a.shape[1])]))
        if c > best[1]:
            best = (k / FPS, c)
    return best


def main():
    include_other = "--include-other" in sys.argv
    idx = pd.read_csv(VICON2D / "vicon2d_index.csv")
    actions = load_actions()
    clips = {c["video"]: c for c in load_clips()}
    align_rows, out_rows = [], []
    for take, views in idx.groupby("take"):
        clip = clips.get(take)
        if clip is None or actions[clip["action_id"]]["role"] == "PRACTICE":
            continue
        kp = KEYPOINT_DIR / f"{take}.csv"
        lag, corr, view = None, None, None
        if kp.exists():
            vt, vs = height_signal(pd.read_csv(kp))
            for name in views["name"]:
                _, ss = height_signal(pd.read_csv(VICON2D / f"{name}.csv"))
                l, c = best_lag(vs, ss)
                if corr is None or c > corr:
                    lag, corr, view = l + vt[0], c, name
        align_rows.append({"take": take, "action_id": clip["action_id"], "best_view": view,
                           "offset_sec": None if lag is None else round(lag, 2),
                           "corr": None if corr is None else round(corr, 3)})
        is_danger = actions[clip["action_id"]]["role"] == "DANGER"
        if not is_danger and not include_other:
            print(f"[건너뜀] {take}: OTHER Take (위험 Take 만 사용, --include-other 로 포함 가능)")
            continue
        s, e = clip["start_sec"], clip["end_sec"]
        if is_danger:
            if lag is None or corr < MIN_CORR or s == "" or e == "":
                print(f"[제외] {take}: 위험 Take 인데 시간 정렬 불확실(corr={corr}) 또는 구간 없음")
                continue
            s, e = round(float(s) - lag, 2), round(float(e) - lag, 2)
        elif s != "" and e != "" and lag is not None and corr >= MIN_CORR:
            s, e = round(float(s) - lag, 2), round(float(e) - lag, 2)
        else:
            s, e = "", ""
        for name in views["name"]:
            out_rows.append({"video": name, "source": "vicon2d", "performer_id": "P001",
                             "action_id": clip["action_id"], "start_sec": s, "end_sec": e,
                             "note": f"VICON 2D 투영 (offset {lag if lag is None else round(lag, 2)}s)"})
        print(f"[{take}] best={view} offset={align_rows[-1]['offset_sec']}s corr={align_rows[-1]['corr']}")

    rep = ensure_dir(REPORT_DIR / "day09")
    pd.DataFrame(align_rows).to_csv(rep / "vicon2d_alignment.csv", index=False, encoding="utf-8-sig")
    p = DATA_DIR / "labels" / "clips_vicon2d.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n[저장] {p} ({len(out_rows)}개), {rep / 'vicon2d_alignment.csv'}")


if __name__ == "__main__":
    main()
