"""바른품 학습 공통 기능 (수업 src/common.py를 바른품 폴더 구조에 맞게 확장)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

# ① 프로젝트 경로 — 이 파일(src/common.py)의 한 단계 위가 training/ 폴더
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIRS = {
    "vicon": DATA_DIR / "raw" / "vicon",   # action_scene.zip 의 C###_T###.mp4
    "phone": DATA_DIR / "raw" / "phone",   # 바탕화면\영상 의 KakaoTalk_*.mp4
    "hmdb51": DATA_DIR / "raw" / "hmdb51", # 외부 데이터 (ext_hmdb51_to_other.py 가 채움)
}
KEYPOINT_DIR = DATA_DIR / "keypoints"      # 5일차 결과: 영상별 Keypoint CSV
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"        # 6일차 결과: 서비스로 복사할 모델 3개


# ② 설정 파일 읽기
def load_json(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config() -> dict:
    """수업과 같은 이름의 함수: configs/project.json"""
    return load_json("project.json")


def load_actions() -> dict[str, dict]:
    """configs/actions.csv → {action_id: {...}}"""
    with (CONFIG_DIR / "actions.csv").open("r", encoding="utf-8-sig") as f:
        return {row["action_id"]: row for row in csv.DictReader(f)}


def load_clips() -> list[dict]:
    """data/labels/clips.csv → 영상별 라벨 목록 (빈 칸은 그대로 문자열 '')"""
    with (DATA_DIR / "labels" / "clips.csv").open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


EXTERNAL_FILES = ["clips_external.csv", "clips_vicon2d.csv"]   # HMDB51, VICON 2D 투영 (둘 다 Train 전용)
TRAIN_ONLY_SOURCES = {"hmdb51", "vicon2d"}


def load_external_clips() -> list[dict]:
    """data/labels/clips_external.csv + clips_vicon2d.csv (없는 파일은 건너뜀)"""
    rows = []
    for name in EXTERNAL_FILES:
        p = DATA_DIR / "labels" / name
        if p.exists():
            with p.open("r", encoding="utf-8-sig") as f:
                rows += list(csv.DictReader(f))
    return rows


def keypoint_path(clip: dict) -> Path:
    """VICON 2D 는 영상이 없고 d09b 가 만든 CSV 가 곧 Keypoint CSV"""
    if clip["source"] == "vicon2d":
        return DATA_DIR / "vicon2d" / f"{clip['video']}.csv"
    return KEYPOINT_DIR / f"{clip['video']}.csv"


def video_path(clip: dict) -> Path:
    return RAW_DIRS[clip["source"]] / f"{clip['video']}.mp4"


# ③ GPU 또는 CPU 선택 (수업 코드 그대로)
def resolve_device(requested: str):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return 0
    return "cpu"


# ④ 폴더 만들기
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
