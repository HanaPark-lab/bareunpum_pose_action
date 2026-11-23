"""6일차 추가 — 사람 돌려가며 평가 (Leave-One-Person-Out, LOPO).

왜?   폰 수행자가 3명(P002·P003·P004)뿐이라 Val 1영상·Test 2영상으로는 숫자가 크게 흔들린다.
      폰 수행자를 한 명씩 '평가용'으로 빼고 나머지 전부로 학습 → 3번 반복 → 평균.
      흔들기 영상 5개가 모두 한 번씩 '처음 보는 사람' 으로 평가된다.
설정:  서비스 모델과 같은 E2 (VICON 영상 + VICON 3D 16각도 + 나머지 폰 수행자), 임계값·반복 수는 train.json 값 고정.
       외부 데이터(hmdb51)는 넣지 않는다.
실행:  uv run python scripts/d06d_lopo_eval.py
결과:  reports/day06/lopo_folds.csv     fold(빠진 사람)별 지표
       reports/day06/lopo_videos.csv    영상별: 위험 창 수, NG(같은 위험 2창 연속) 여부
읽는 법: 평균 ± 최소~최대 를 같이 적는다. 이 값은 '모델 선택'에 쓰지 않는다 (선택은 이미 Val 로 끝남).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.common import DATA_DIR, REPORT_DIR, ensure_dir, load_json

DANGER = ["shake", "drop", "look_away"]
USE_SOURCES = {"vicon", "phone", "vicon2d"}   # = E2


def video_ng(df: pd.DataFrame, thr: float, repeat: int) -> str | None:
    """영상 1개의 창들을 시간순으로 보고, 같은 위험이 repeat 번 연속 ≥ thr 이면 그 위험 이름."""
    streak_label, streak = None, 0
    for _, r in df.sort_values("start_sec").iterrows():
        probs = {c: r[f"p_{c}"] for c in DANGER if f"p_{c}" in r}
        best = max(probs, key=probs.get)
        if probs[best] >= thr:
            streak = streak + 1 if streak_label == best else 1
            streak_label = best
            if streak >= repeat:
                return best
        else:
            streak_label, streak = None, 0
    return None


def main():
    cfg = load_json("train.json")
    thr, repeat = cfg["danger_threshold"], cfg["repeat_windows"]
    d = np.load(DATA_DIR / "windows" / "windows.npz", allow_pickle=True)
    X, y = d["X"], d["y"].astype(str)
    meta = pd.read_csv(DATA_DIR / "windows" / "windows_meta.csv")
    use = meta["source"].isin(USE_SOURCES).to_numpy()
    phone_people = sorted(meta.loc[meta["source"] == "phone", "performer_id"].dropna().unique())
    print(f"[설정] E2 소스={sorted(USE_SOURCES)}  thr={thr}  repeat={repeat}  폰 수행자={phone_people}")

    folds, vids = [], []
    for p in phone_people:
        te = (meta["performer_id"] == p).to_numpy() & (meta["source"] == "phone").to_numpy()
        tr = use & ~te
        rf = RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                    class_weight=cfg.get("class_weight"), random_state=cfg["random_state"],
                                    n_jobs=-1).fit(X[tr], y[tr])
        P = rf.predict_proba(X[te])
        cls = list(rf.classes_)
        part = meta[te].copy()
        for c in DANGER:
            part[f"p_{c}"] = P[:, cls.index(c)] if c in cls else 0.0
        yy = y[te]
        alarm = part[[f"p_{c}" for c in DANGER]].max(axis=1).to_numpy() >= thr
        row = {"held_out": p, "windows": len(yy), "OTHER_windows": int((yy == "OTHER").sum()),
               "other_false_alarm@thr": round(float(alarm[yy == "OTHER"].mean()), 3) if (yy == "OTHER").any() else None}
        for c in DANGER:
            m = yy == c
            row[f"{c}_windows"] = int(m.sum())
            row[f"{c}_recall@thr"] = round(float((part.loc[m, f"p_{c}"] >= thr).mean()), 3) if m.any() else None
        for v, g in part.groupby("video"):
            label = g["label"].mode().iat[0] if (g["label"] == "OTHER").all() else g.loc[g["label"] != "OTHER", "label"].iat[0]
            ng = video_ng(g, thr, repeat)
            vids.append({"held_out": p, "video": v, "video_label": label, "windows": len(g),
                         "alarm_windows": int((g[[f"p_{c}" for c in DANGER]].max(axis=1) >= thr).sum()),
                         "NG": ng or "", "correct": (ng == label) if label != "OTHER" else (ng is None)})
        folds.append(row)
        print(f"[fold {p}] " + "  ".join(f"{k}={v}" for k, v in row.items() if k != "held_out"))

    fd = pd.DataFrame(folds)
    vd = pd.DataFrame(vids)
    rep = ensure_dir(REPORT_DIR / "day06")
    fd.to_csv(rep / "lopo_folds.csv", index=False, encoding="utf-8-sig")
    vd.to_csv(rep / "lopo_videos.csv", index=False, encoding="utf-8-sig")

    print("\n=== LOPO 요약 (폰 수행자 한 명씩 평가) ===")
    for col in ["other_false_alarm@thr"] + [f"{c}_recall@thr" for c in DANGER]:
        s = fd[col].dropna()
        if len(s):
            print(f"{col:24s} 평균 {s.mean():.3f}  (최소 {s.min():.3f} ~ 최대 {s.max():.3f}, fold {len(s)}개)")
    # 창 수로 가중한 전체 값 (fold 마다 창 수가 달라서 같이 본다)
    sh = vd[vd["video_label"] == "shake"]
    ot = vd[vd["video_label"] == "OTHER"]
    print(f"\n[영상 기준] 흔들기 영상 {len(sh)}개 중 NG(shake) {int(sh['correct'].sum())}개  |  "
          f"정상 영상 {len(ot)}개 중 NG 오경보 {int((ot['NG'] != '').sum())}개")
    print(vd.to_string(index=False))
    print(f"\n[저장] {rep / 'lopo_folds.csv'}, {rep / 'lopo_videos.csv'}")


if __name__ == "__main__":
    main()
