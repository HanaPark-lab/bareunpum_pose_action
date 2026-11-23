"""외부 데이터 — HMDB51 (CC BY 4.0) 일부 동작을 OTHER(일반 동작) Train 보강용으로 준비.

출처: Kuehne et al., "HMDB: A Large Video Database for Human Motion Recognition", ICCV 2011.
      https://serre.lab.brown.edu/hmdb51.html  (License: CC BY 4.0) → README 에 출처 표기 필수

준비:  공식 페이지에서 hmdb51_org.rar 를 받아 풀면 동작별 .rar 가 또 있음 → 모두 풀어서
       data/external/hmdb51/<동작폴더>/*.avi  형태로 둔다 (예: data/external/hmdb51/walk/xxx.avi)
실행:  uv run python scripts/ext_hmdb51_to_other.py --per-class 20
       → data/raw/hmdb51/ 로 복사 + data/labels/clips_external.csv 생성
       이어서  d04_extract_keypoints.py --external  →  d05  →  d06 (E1x 가 자동으로 비교됨)
규칙:  - 항상 Train 에만 (Val/Test 에 넣지 않음)
       - fall_floor, throw 는 '아기 떨구기'가 아니므로 drop 라벨로 쓰지 않음 (아예 제외)
       - 사람이 여러 명이거나 화면이 작은 클립이 많으니 d04 요약의 detect_ratio 로 걸러 낸다
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import DATA_DIR, ensure_dir

# 폴더 이름은 HMDB51 압축 해제 결과 기준 (공식 클래스명: walk, turn, pick, sit down → sit, ...)
USE = ["walk", "turn", "pick", "sit", "stand", "hug", "wave", "drink", "catch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA_DIR / "external" / "hmdb51"))
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} 가 없습니다. 위 '준비' 설명대로 HMDB51 을 풀어 주세요.")
    raw = ensure_dir(DATA_DIR / "raw" / "hmdb51")
    rnd = random.Random(args.seed)
    rows = []
    for cls in USE:
        files = sorted((root / cls).glob("*.avi"))
        if not files:
            print(f"[없음] {cls}")
            continue
        for src in rnd.sample(files, min(args.per_class, len(files))):
            name = f"HMDB_{cls}_{src.stem}"[:120]
            dst = raw / f"{name}.mp4"
            if not dst.exists():
                shutil.copy(src, dst)   # 확장자만 .mp4 (OpenCV 는 내용으로 읽음)
            # HMDB51 파일명에는 원본 영상 이름이 들어 있음 → 같은 원본은 같은 performer 로 묶음
            rows.append({"video": name, "source": "hmdb51", "performer_id": "HMDB_" + src.stem.split("_")[0][:40],
                         "action_id": "EXT_OTHER", "start_sec": "", "end_sec": "", "note": f"HMDB51 {cls} (CC BY 4.0)"})
        print(f"[{cls}] {min(args.per_class, len(files))}개")
    out = DATA_DIR / "labels" / "clips_external.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[저장] {out} ({len(rows)}개)")


if __name__ == "__main__":
    main()
