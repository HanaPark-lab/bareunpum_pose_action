"""6일차 추가 — 외부 데이터(HMDB51, CC BY 4.0)로 '평가만' 하기. 학습에는 절대 넣지 않는다.

출처: Kuehne et al., "HMDB: A Large Video Database for Human Motion Recognition", ICCV 2011.
      https://huggingface.co/datasets/Serrelab/hmdb51  (hmdb51_org.rar, License: CC BY 4.0)

무엇을 재나?
  ① 오경보: 일상 동작(걷기·줍기·안기·앉기 …) 영상에서 NG 가 얼마나 잘못 뜨는지
     - 창 기준: 위험 확률 ≥ 임계값인 창 비율 (d06 의 other_false_alarm@thr 와 같은 뜻)
     - 영상 기준: 서비스 규칙(같은 위험 2창 연속)으로 NG 가 뜬 영상 비율
  ② 허리 규칙: pick(줍기) 영상을 LIFT 모드로 돌려 CAUTION / OK 를 기록하고,
     몸통이 가장 많이 숙여진 순간의 사진을 저장 → 사람이 '무릎 펴고 숙임 / 쪼그려 앉음' 을 보고 맞는지 확인

준비:  data/external/hmdb51/<동작폴더>/*.avi   (README 의 HMDB51 받기 참고)
실행:  uv run python scripts/d06c_external_test.py                 # 동작당 20개
       uv run python scripts/d06c_external_test.py --per-class 40
결과:  reports/day06c/external_clips.csv        영상별 결과 (GENERAL)
       reports/day06c/external_summary.csv      동작별 오경보율
       reports/day06c/pick_lift.csv             pick 영상 LIFT 모드 결과 (+ 사람이 채울 posture 칸)
       reports/day06c/pick_frames/<영상>.jpg    몸통이 가장 숙여진 Frame (관절·각도 표시)
주의:  HMDB51 은 영화·유튜브 장면이라 화질이 낮고(240p) 아기가 없다 → '처음 보는 환경에서의 오경보' 참고 지표.
       E1x(HMDB51 로 학습한 모델)를 서비스 모델로 쓰면 이 평가는 누수가 되므로 하지 않는다.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from src.common import DATA_DIR, MODEL_DIR, REPORT_DIR, ensure_dir, load_json
from src.pose_extract import load_pose_model
from src.realtime_engine import StreamAnalyzer, load_rf

HMDB_DIR = DATA_DIR / "external" / "hmdb51"
# 아기 돌봄과 비슷한 '정상' 일상 동작만. fall_floor·throw·punch 같은 동작은 OTHER 가 아니므로 뺀다.
CLASSES = ["walk", "turn", "pick", "sit", "stand", "hug", "wave", "drink", "catch", "pour", "eat", "talk"]


def reset_tracker(model):
    if getattr(model, "predictor", None) is not None and getattr(model.predictor, "trackers", None):
        for t in model.predictor.trackers:
            t.reset()


def run_clip(path: Path, pose, pcfg, rf, info, rule, mode: str, keep_frame: bool = False):
    """영상 1개 → (수행자 요약, 창 결과 목록, 몸통이 가장 숙여진 Frame)"""
    analyzer = StreamAnalyzer(pose, pcfg, rf, info, rule, mode=mode, main_only=True)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n, best = 0, (-1.0, None)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r, _ = analyzer.feed(frame, n / fps)
        if keep_frame:
            res = analyzer.latest.get(analyzer.main_track())
            lean = res.get("trunk_lean_deg") if res else None
            if lean is not None and lean > best[0]:
                best = (float(lean), r.plot())
        n += 1
    cap.release()
    reset_tracker(pose)
    main = analyzer.main_track()
    summary = next((s for s in analyzer.summary() if s["is_main"]), None)
    windows = analyzer.results.get(main, []) if main is not None else []
    return summary, windows, best, n / fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not HMDB_DIR.exists():
        raise SystemExit(f"{HMDB_DIR} 가 없습니다. HMDB51 을 받아 동작 폴더별로 풀어 두세요.")

    rf, _, info = load_rf(MODEL_DIR)
    thr = info["danger_threshold"]
    rule = load_json("waist_rule.json")
    pose, pcfg = load_pose_model()
    rep = ensure_dir(REPORT_DIR / "day06c")
    frame_dir = ensure_dir(rep / "pick_frames")
    rng = random.Random(args.seed)
    print(f"[모델] {info['model_version']}  thr={thr}  repeat={info['repeat_windows']}")

    rows, pick_rows = [], []
    for cls in CLASSES:
        vids = sorted((HMDB_DIR / cls).glob("*.avi"))
        if not vids:
            print(f"[건너뜀] {cls}: 영상 없음")
            continue
        vids = rng.sample(vids, min(args.per_class, len(vids)))
        for v in vids:
            s, wins, _, dur = run_clip(v, pose, pcfg, rf, info, rule, "GENERAL")
            probs = [w["danger_prob"] for w in wins if w.get("danger_prob") is not None]
            rows.append({"class": cls, "video": v.name, "seconds": round(dur, 1),
                         "result": s["result"] if s else "NO_PERSON", "reason": s["reason"] if s else None,
                         "windows": len(probs), "alarm_windows": int(sum(p >= thr for p in probs)),
                         "max_danger_prob": round(max(probs), 3) if probs else None,
                         "top_danger": max(wins, key=lambda w: w.get("danger_prob") or 0).get("danger_label") if probs else None})
            if cls == "pick":
                s2, wins2, (lean, img), _ = run_clip(v, pose, pcfg, rf, info, rule, "LIFT", keep_frame=True)
                if img is not None:
                    cv2.putText(img, f"max lean {lean:.0f}  -> {s2['result'] if s2 else '-'}", (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imwrite(str(frame_dir / f"{v.stem}.jpg"), img)
                pick_rows.append({"video": v.name, "lift_result": s2["result"] if s2 else "NO_PERSON",
                                  "reason": s2["reason"] if s2 else None,
                                  "max_trunk_lean_deg": round(lean, 1) if lean >= 0 else None,
                                  "caution_windows": sum(w["result"] == "CAUTION" for w in wins2),
                                  "posture": ""})   # 사람이 사진을 보고 stoop(무릎 펴고 숙임) / squat / unclear 로 채움
        done = [r for r in rows if r["class"] == cls]
        print(f"[{cls:6s}] {len(done):3d}개  NG {sum(r['result'] == 'NG' for r in done)}")

    df = pd.DataFrame(rows)
    df.to_csv(rep / "external_clips.csv", index=False, encoding="utf-8-sig")
    valid = df[df["windows"] > 0]
    summ = valid.groupby("class").agg(clips=("video", "size"), ng_clips=("result", lambda x: int((x == "NG").sum())),
                                      windows=("windows", "sum"), alarm_windows=("alarm_windows", "sum"))
    summ.loc["ALL"] = summ.sum()
    summ["clip_ng_rate"] = (summ["ng_clips"] / summ["clips"]).round(3)
    summ["window_false_alarm@thr"] = (summ["alarm_windows"] / summ["windows"]).round(3)
    summ.to_csv(rep / "external_summary.csv", encoding="utf-8-sig")
    if pick_rows:
        pd.DataFrame(pick_rows).to_csv(rep / "pick_lift.csv", index=False, encoding="utf-8-sig")

    print("\n=== 외부 Test (HMDB51, 모두 정상 동작) ===")
    print(summ.to_string())
    print(f"\n사람 없음/창 부족으로 제외: {len(df) - len(valid)}개")
    ng = df[df["result"] == "NG"]
    if len(ng):
        print("\n[NG 가 뜬 영상] → 영상을 직접 열어 왜 위험처럼 보였는지 확인")
        print(ng[["class", "video", "reason", "max_danger_prob"]].to_string(index=False))
    if pick_rows:
        p = pd.DataFrame(pick_rows)
        print(f"\n=== pick LIFT 모드: CAUTION {int((p.lift_result == 'CAUTION').sum())} / OK {int((p.lift_result == 'OK').sum())}"
              f" / UNKNOWN {int((p.lift_result == 'UNKNOWN').sum())} ===")
        print(f"→ {frame_dir} 사진을 보고 pick_lift.csv 의 posture 칸에 stoop / squat / unclear 를 적은 뒤 "
              f"'stoop→CAUTION, squat→OK' 비율을 계산")
    print(f"\n[저장] {rep}")


if __name__ == "__main__":
    main()
