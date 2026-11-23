"""9일차 확장 — VICON 3D(FBX) → 여러 가상 카메라 각도의 2D COCO 17 Keypoint CSV (Blender 로 실행).

실행 (Windows PowerShell 또는 WSL, Blender 설치 폴더의 blender.exe 사용):
  "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" -b -P scripts/d09b_vicon_to_2d.py -- ^
      --fbx-dir data/external/Motion_VIsol --out-dir data/vicon2d
  (WSL 에서는 경로를 /mnt/c/... 로, 줄바꿈 ^ 대신 \\)
옵션:  --yaws 0 45 90 135 180 225 270 315   --heights 1.0 1.6   --distance 4.5   --fps 30
결과:  data/vicon2d/<C###_T###>_y<yaw>_h<height>.csv  (d04 Keypoint CSV 와 같은 열 → d05 가 그대로 읽음)
       data/vicon2d/vicon2d_index.csv   (take, cam, 파일, 프레임 수)
원리:  렌더링 없이 관절 3D 좌표를 핀홀 카메라(1920×1080, 초점 1400px)로 투영만 한다 → 빠름.
       얼굴 5점(코·눈·귀)은 VICON 에 없어 Head 뼈 기준 고정 오프셋으로 근사.
       conf 는 모두 0.95 (가림 없음) → 실제 YOLO 보다 깨끗하므로 d09 jitter 와 같이 쓰는 걸 권장.
주의:  같은 Take 의 RGB 영상과 함께 Train 에만 넣는다 (P001).
"""
import argparse
import csv
import sys
from pathlib import Path

import bpy
import numpy as np

COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
BONE = {  # COCO 관절 → VICON 뼈의 시작점(head)
    "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
    "left_hip": "LeftUpLeg", "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg", "right_knee": "RightLeg",
    "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
}
# 얼굴 근사: Head 뼈 원점 기준 (forward, left, up) 미터
FACE = {"nose": (0.10, 0.0, 0.06), "left_eye": (0.08, 0.035, 0.09), "right_eye": (0.08, -0.035, 0.09),
        "left_ear": (0.0, 0.075, 0.07), "right_ear": (0.0, -0.075, 0.07)}
W, H, FOCAL = 1920, 1080, 1400.0


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--yaws", nargs="*", type=float, default=[0, 45, 90, 135, 180, 225, 270, 315])
    ap.add_argument("--heights", nargs="*", type=float, default=[1.0, 1.6])
    ap.add_argument("--distance", type=float, default=4.5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--only", nargs="*", default=None, help="예: C003_T001 C012_T001")
    return ap.parse_args(argv)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def load_take(fbx: Path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(fbx))
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    prefix = arm.pose.bones[0].name.split(":")[0] + ":" if ":" in arm.pose.bones[0].name else ""
    act = arm.animation_data.action
    f0, f1 = int(act.frame_range[0]), int(act.frame_range[1])
    src_fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    return arm, prefix, f0, f1, src_fps


def joints_at(arm, prefix, frame, head_axes):
    bpy.context.scene.frame_set(frame)
    mw = arm.matrix_world
    P = {}
    for c, b in BONE.items():
        P[c] = np.array(mw @ arm.pose.bones[prefix + b].head)
    hb = arm.pose.bones[prefix + "Head"]
    hpos = np.array(mw @ hb.head)
    m = np.array((mw @ hb.matrix).to_3x3())
    m = m / np.linalg.norm(m, axis=0, keepdims=True)
    fwd, left, up = (m[:, i] * s for i, s in head_axes)
    for c, (a, b, d) in FACE.items():
        P[c] = hpos + a * fwd + b * left + d * up
    return np.stack([P[c] for c in COCO])  # (17,3) 미터, Z 위쪽


def head_axes_from_tpose(arm, prefix, frame):
    """첫 프레임(T자)에서 Head 뼈의 어느 축이 앞/왼쪽/위인지 자동으로 찾기"""
    bpy.context.scene.frame_set(frame)
    mw = arm.matrix_world
    ls = np.array(mw @ arm.pose.bones[prefix + "LeftArm"].head)
    rs = np.array(mw @ arm.pose.bones[prefix + "RightArm"].head)
    left_w = unit(ls - rs)
    up_w = np.array([0.0, 0.0, 1.0])
    fwd_w = unit(np.cross(left_w, up_w))
    m = np.array((mw @ arm.pose.bones[prefix + "Head"].matrix).to_3x3())
    m = m / np.linalg.norm(m, axis=0, keepdims=True)
    axes = []
    for target in (fwd_w, left_w, up_w):
        dots = [float(np.dot(m[:, i], target)) for i in range(3)]
        i = int(np.argmax(np.abs(dots)))
        axes.append((i, 1.0 if dots[i] > 0 else -1.0))
    return axes


def project(points, cam_pos, target):
    fwd = unit(target - cam_pos)
    right = unit(np.cross(fwd, np.array([0.0, 0.0, 1.0])))
    down = np.cross(fwd, right)
    rel = points - cam_pos
    z = rel @ fwd
    x = (rel @ right) / z * FOCAL + W / 2
    y = (rel @ down) / z * FOCAL + H / 2
    return np.stack([x, y], axis=1)


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for fbx in sorted(Path(args.fbx_dir).glob("*_LSJ.fbx")):
        take = "_".join(fbx.stem.split("_")[1:3])      # 261111_C003_T001_LSJ → C003_T001
        if args.only and take not in args.only:
            continue
        arm, prefix, f0, f1, src_fps = load_take(fbx)
        axes = head_axes_from_tpose(arm, prefix, f0)
        n_out = int((f1 - f0) / src_fps * args.fps) + 1
        seq = np.stack([joints_at(arm, prefix, f0 + int(round(k / args.fps * src_fps)), axes) for k in range(n_out)])
        center = seq[:, 11:13].mean(axis=(0, 1))       # Take 전체 골반 평균 위치를 카메라가 바라봄
        for yaw in args.yaws:
            for h in args.heights:
                r = np.radians(yaw)
                cam = np.array([center[0] + args.distance * np.cos(r), center[1] + args.distance * np.sin(r), h])
                tgt = np.array([center[0], center[1], 0.9])
                name = f"{take}_y{int(yaw):03d}_h{int(h * 100):03d}"
                rows = []
                for k in range(n_out):
                    xy = project(seq[k], cam, tgt)
                    row = {"frame_id": k, "timestamp_sec": round(k / args.fps, 4), "track_id": 1,
                           "pose_detected": 1, "person_count": 1, "box_conf": 0.95}
                    for j, c in enumerate(COCO):
                        row[f"{c}_x_px"], row[f"{c}_y_px"] = round(float(xy[j, 0]), 2), round(float(xy[j, 1]), 2)
                        row[f"{c}_conf"] = 0.95
                    rows.append(row)
                with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                index.append({"take": take, "name": name, "yaw": yaw, "height": h, "frames": n_out,
                              "seconds": round(n_out / args.fps, 2)})
        print(f"[{take}] {n_out} frames × {len(args.yaws) * len(args.heights)} 시점")
    with (out / "vicon2d_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    print(f"[저장] {out} ({len(index)}개 CSV)")


main()
