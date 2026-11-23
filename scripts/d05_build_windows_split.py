"""5일차 — 라벨·구간으로 2초 창 Dataset 만들기 + 사람 단위 Split + 누수 검사.

실행:  uv run python scripts/d05_build_windows_split.py
입력:  data/keypoints/*.csv (d04), data/labels/clips.csv, configs/{actions.csv, split.json, train.json}
       (있으면) data/labels/clips_external.csv  ← ext_hmdb51_to_other.py 결과, Train 전용
       (있으면) data/labels/clips_vicon2d.csv   ← d09b·d09c 결과 (VICON 3D → 16시점 2D), Train 전용
결과:  data/windows/windows.npz   X(N, 1140), y(N,)
       data/windows/windows_meta.csv  창마다 video, performer, split, label, start, end, source
       reports/day05/split_summary.csv  split × label 창 개수
확인:  ① 누수 검사 PASS (사람·영상이 두 split에 걸치지 않음)
       ② Val/Test 에 위험 3종(shake/drop/look_away)이 각각 몇 개인지 — 0이면 그 클래스는 평가 불가
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common import DATA_DIR, REPORT_DIR, ensure_dir, load_actions, load_clips, load_external_clips, load_json, keypoint_path, TRAIN_ONLY_SOURCES
from src.sequence_utils import sequence_feature_names
from src.window_utils import build_clip_windows


def to_float(v):
    return None if v in ("", None) else float(v)


def main():
    cfg = load_json("train.json")
    split_cfg = load_json("split.json")
    actions = load_actions()
    performer_split = {p: s for s in ("train", "val", "test") for p in split_cfg[s]}

    clips = load_clips() + load_external_clips()
    X, meta = [], []
    skipped = Counter()
    for clip in clips:
        aid = clip["action_id"]
        if aid == "":
            skipped["라벨 없음"] += 1
            continue
        if aid == "EXT_OTHER":
            base = "OTHER"
        else:
            act = actions.get(aid)
            if act is None:
                skipped[f"모르는 action {aid}"] += 1
                continue
            if act["role"] == "PRACTICE":
                skipped["연습 모드(CPR·하임리히) 제외"] += 1
                continue
            base = act["rf_label"]
        performer = clip.get("performer_id", "")
        split = "train" if clip["source"] in TRAIN_ONLY_SOURCES else performer_split.get(performer)
        if split is None:
            skipped[f"split 미배정 performer '{performer}'"] += 1
            continue
        kp_path = keypoint_path(clip)
        if not kp_path.exists():
            skipped["Keypoint CSV 없음 (d04 먼저)"] += 1
            continue
        seg_s, seg_e = to_float(clip.get("start_sec")), to_float(clip.get("end_sec"))
        if base != "OTHER" and (seg_s is None or seg_e is None):
            skipped[f"위험 영상인데 구간 없음: {clip['video']}"] += 1
            continue
        for vec, label, s, e in build_clip_windows(pd.read_csv(kp_path), base, seg_s, seg_e, cfg):
            X.append(vec)
            meta.append({"video": clip["video"], "source": clip["source"], "performer_id": performer,
                         "action_id": aid, "split": split, "label": label,
                         "start_sec": round(s, 2), "end_sec": round(e, 2)})

    for k, v in skipped.items():
        print(f"[제외] {k}: {v}개 영상")
    if not X:
        raise SystemExit("만들어진 창이 없습니다. clips.csv·split.json·d04 결과를 확인하세요.")

    out = ensure_dir(DATA_DIR / "windows")
    X = np.stack(X).astype(np.float32)
    meta_df = pd.DataFrame(meta)
    names = sequence_feature_names(cfg["sequence_steps"])
    assert X.shape[1] == len(names), f"Feature 수 불일치 {X.shape[1]} != {len(names)}"
    np.savez_compressed(out / "windows.npz", X=X, y=meta_df["label"].to_numpy())
    meta_df.to_csv(out / "windows_meta.csv", index=False, encoding="utf-8-sig")

    # 누수 검사: 한 사람·한 영상은 한 split 에만
    leak_p = meta_df.groupby("performer_id")["split"].nunique()
    leak_v = meta_df.groupby("video")["split"].nunique()
    bad = list(leak_p[leak_p > 1].index) + list(leak_v[leak_v > 1].index)
    print(f"\n[누수 검사] {'PASS' if not bad else 'FAIL ' + str(bad)}")

    table = meta_df.pivot_table(index="label", columns="split", values="video", aggfunc="count", fill_value=0)
    print(f"\n[창 개수] X={X.shape}\n{table}")
    rep = ensure_dir(REPORT_DIR / "day05")
    table.to_csv(rep / "split_summary.csv", encoding="utf-8-sig")
    for sp in ("val", "test"):
        if sp in table.columns:
            missing = [c for c in ("shake", "drop", "look_away") if table.get(sp, pd.Series()).get(c, 0) == 0]
            if missing:
                print(f"[주의] {sp} 에 {missing} 창이 없습니다 → 그 클래스 Recall은 평가 불가")
    print(f"\n[저장] {out / 'windows.npz'}, {out / 'windows_meta.csv'}")


if __name__ == "__main__":
    main()
