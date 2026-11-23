# 바른품 동작 인식 — 교과10 1~8일차를 바른품 데이터로 (+ 9일차 보강)

> 결과물: `models/` 의 3개 파일 → 자율 프로젝트 저장소(`bareunpum_platform`)의 `ai_service/models/` 로 복사
> 9일차(d09·d09b·d09c·HMDB51)는 선택 확장

## 0. 준비 (15분)

```bash
# WSL — 저장소 폴더(bareunpum_pose_action)에서
uv venv && source .venv/bin/activate
uv pip install -e .          # 또는: uv pip install ultralytics opencv-python pandas scikit-learn joblib matplotlib lap
```

Blender (d09b 용): Windows 의 `blender.exe` 를 그대로 써도 되고, WSL 에서는
`"/mnt/c/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b -P ...` 처럼 부른다 (경로는 설치 버전에 맞게).
FBX: `Downloads\Motion_VIsol.zip` 을 풀어 `data/external/Motion_VIsol/` 에 `*_LSJ.fbx` 가 오게 둔다.

영상 복사 (Windows 경로 → WSL):

```bash
# VICON 영상 20개 (Downloads\action_scene.zip 을 푼 것)
unzip /mnt/c/Users/user1/Downloads/action_scene.zip -d /tmp/ && cp /tmp/action_scene/*.mp4 data/raw/vicon/
# 휴대폰 영상 12개
cp /mnt/c/Users/user1/Desktop/영상/*.mp4 data/raw/phone/
```

라벨 2개 채우기:
- `data/labels/clips.csv` — VICON 행은 `start_sec,end_sec`(핵심 동작 구간), 휴대폰 행은 `performer_id,action_id,start_sec,end_sec`
  - 위험 영상(A003 흔들기, A005 다른 곳 보기, A012 떨구기)은 **구간 필수**. OTHER 는 비워 두면 영상 전체
  - 썸네일: `Downloads\_bareunpum_check\thumbs\`(VICON), `thumbs_kakao\`(휴대폰)
- `configs/split.json` — 휴대폰 촬영자(P002~)를 `val`/`test` 에 사람 단위로 배정. P001(VICON)=train

## 1. 실행 순서와 확인할 것

| 일차 | 명령 | 볼 것 |
|---|---|---|
| 1 | `python scripts/d01_env_and_pose_video.py --video C003_T001` | `reports/day01/C003_T001_pose.mp4` 에 수행자·뒤 사람 모두 Skeleton |
| 2 | `python scripts/d02_pose_quality.py` | `pose_quality.csv`: 수행자 검출률, 하체 conf (허리 규칙 가능 여부) |
| 3 | `python scripts/d03_frame_features.py --video C010_T001` | `C010_T001_waist.png`: 쪼그려 앉기 = 몸통↑ 무릎↓ → 위반 아님 |
|   | `python scripts/d03_frame_features.py --video C004_T001` | 물건 줍기 때 무릎 펴고 숙이는지 |
| 4·5 | `python scripts/d04_extract_keypoints.py` | `extract_summary.csv`: VICON 은 사람 6~14명 중 수행자 1명이 끝까지 같은 ID |
| 5 | `python scripts/d05_build_windows_split.py` | 누수 검사 PASS, split × label 창 개수 (Val 에 위험 3종이 있는지) |
| 6 | `python scripts/d06_train_rf.py` | `val_metrics.csv`: E0 vs E1, `other_false_alarm@thr`, 위험별 `recall@thr` → `models/` 3개 저장 |
| 7 | `python scripts/d07_realtime_decision.py --video <Val 영상>` | `_decision.mp4`: 위험 구간에서만 NG, 들어 올릴 때 오경보 여부 |
|   | `... --video C004_T001 --mode LIFT` | 허리 CAUTION |
|   | `... --url http://<폰IP>:8080/video --seconds 8` | IP Webcam 8초 → 내일 서비스와 같은 입력 |
| 8 | `python scripts/d08_multi_person.py --video C012_T001` | 수행자 ID 변경 0회, 뒤에 앉은 사람 판정 (잠깐 보인 사람의 오경보 확인) |
| 9 | `python scripts/d09_augment_compare.py` | flip·jitter·speed 보강 전후 같은 Val 비교. 좋으면 `--save` |
| 9+ | `blender -b -P scripts/d09b_vicon_to_2d.py -- --fbx-dir data/external/Motion_VIsol --out-dir data/vicon2d` | Take당 16시점(8방향×높이 1.0·1.6m) 2D CSV. 20 Take 약 1분 |
|   | `python scripts/d09c_align_vicon2d.py` | `vicon2d_alignment.csv`: 영상↔FBX 시간차, corr ≥ 0.6 (위험 Take 는 미달 시 제외) |
|   | `python scripts/d05_build_windows_split.py` → `python scripts/d06_train_rf.py` | **E2(VICON 2D 추가)** 가 E1 보다 Val 에서 나은지. 좋으면 `--save E2` |
| 외부 | `python scripts/ext_hmdb51_to_other.py` → `d04 --external` → `d05` → `d06` | E1x(HMDB51 OTHER 추가)가 오경보를 줄이는지. 둘 다 있으면 E3 |

## 2. 설계 요약 (수업과 다른 점)

| 항목 | 수업 | 바른품 | 이유 |
|---|---|---|---|
| 샘플 | 클립 1개 = 1개 | **2초 창(0.5초 간격)** = 1개 | 영상 하나에 안아 올리기·위험 동작·내려놓기가 섞여 있음 |
| Feature | 30 Step × 38 = 1,140 | **같음** (수업 `sequence_utils` 그대로) | 서비스·실시간과 형식 통일 |
| 사람 선택 | 가장 큰 박스 | **ByteTrack + 누적 면적 최대 Track** | VICON 영상 뒤에 4~6명 앉아 있음 |
| 클래스 | 6동작 | **OTHER + shake/drop/look_away** | C안. CPR·하임리히는 연습 모드라 제외 |
| 구간 밖 창 | — | 위험 영상의 구간 밖 = OTHER, 경계 ±0.5초 제외 | 안아 올리기·내려놓기는 일반 돌봄 |
| 판정 | 예측 Label | **NG(0.70 이상 2창 연속) > CAUTION(LIFT 모드 허리) > OK / UNKNOWN** | C안 §4.4 |
| Split | S01/S02/S03 | P001(VICON)=Train, 휴대폰 촬영자 = Val/Test | 사람 단위 누수 방지 |

## 3. 파일

```
bareunpum_pose_action/
├── configs/  project.json · actions.csv(13동작→역할) · train.json · waist_rule.json · split.json
├── src/      common · pose_utils* · feature_utils* · sequence_utils* · augmentation_utils*   (*수업 코드 그대로)
│             pose_extract (Tracking·수행자 선택) · waist_utils (허리 규칙) · window_utils (2초 창)
│             decision_utils (OK/CAUTION/NG/UNKNOWN) · realtime_engine (7·8일차, 내일 FastAPI 재사용)
├── scripts/  d01~d09, d09b(Blender 투영)·d09c(시간 맞춤), ext_hmdb51_to_other
├── data/     raw/{vicon,phone,hmdb51} · external/Motion_VIsol · vicon2d/ · labels/clips*.csv · keypoints/ · windows/
├── models/   rf_action_baseline.joblib · rf_feature_columns.json · rf_training_info.json
└── reports/  dayNN/
```

## 4. 한계 (발표·README 에 그대로)

- 학습 위험 영상이 1~2개(수행자 1명)뿐 → Val/Test 수치는 참고용. 같은 사람 GroupKFold 결과는 낙관적
- 떨구기 순간은 약 2초 → 2초 창 1~3개뿐
- 성인 2D 관절만으로 실제 낙상·시선·아기 상태를 확정할 수 없음 → '패턴 의심 감지'로 표현
- 45° / 160° / 30% / 0.70 은 검증 전 초기값. 몸통 기울기는 화면상 2D 각도
- HMDB51 (CC BY 4.0) 사용 시 출처 표기: Kuehne et al., ICCV 2011
- VICON 2D 는 같은 사람·같은 Take 의 다른 시점일 뿐 새 사람이 아님. 얼굴 5점은 Head 뼈로 근사, conf=0.95 고정(가림 없음)
  → 실제 YOLO 보다 깨끗하므로 Val 비교로 효과를 확인하고, 좋아지지 않으면 쓰지 않는다
