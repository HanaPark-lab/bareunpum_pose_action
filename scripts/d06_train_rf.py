"""6일차 — Random Forest 학습·Val 평가·모델 3개 저장 (수업 35·36 대응).

실험 (C안 §6):
  E0  위험 3종만 (shake/drop/look_away)       → 비교 기준. 정상 동작도 무조건 위험 중 하나로 예측함
  E1  OTHER + 위험 3종                         → 기본 후보
  E1x E1 + 외부 데이터(HMDB51, Train 전용)     → clips_external.csv 가 있을 때만
  E2  E1 + VICON 3D → 16시점 2D (Train 전용)   → clips_vicon2d.csv 가 있을 때만 (d09b·d09c)
  E3  E1 + HMDB51 + VICON 2D                   → 둘 다 있을 때

실행:  uv run python scripts/d06_train_rf.py                 # 모든 실험 비교 + E1 저장
       uv run python scripts/d06_train_rf.py --save E1x      # E1x 를 서비스 모델로 저장
결과:  models/rf_action_baseline.joblib, rf_feature_columns.json, rf_training_info.json
       reports/day06/val_metrics.csv, reports/day06/val_confusion_<exp>.csv
판정 지표:
  - danger_recall@thr : 위험 창 중 '해당 위험 확률 ≥ 0.70' 으로 맞힌 비율
  - other_false_alarm@thr : OTHER 창 중 '아무 위험 확률 ≥ 0.70' 이 나온 비율 (낮을수록 좋음)
Val 이 비어 있으면(카톡 라벨 전) Train 안에서 '영상 단위 GroupKFold' 로 대신 평가한다 (참고용).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

from src.common import DATA_DIR, MODEL_DIR, REPORT_DIR, ensure_dir, load_json
from src.sequence_utils import sequence_feature_names

DANGER = ["shake", "drop", "look_away"]


def make_rf(cfg):
    return RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                                  class_weight=cfg.get("class_weight"), random_state=cfg["random_state"],
                                  n_jobs=-1)


EXTRA = {"E0": set(), "E1": set(), "E1x": {"hmdb51"}, "E2": {"vicon2d"}, "E3": {"hmdb51", "vicon2d"}}
EXTRA_SOURCES = {"hmdb51", "vicon2d"}


def exp_mask(meta: pd.DataFrame, exp: str) -> np.ndarray:
    """실험별로 학습에 쓸 창. 기본(VICON 영상·휴대폰) + 실험이 허용한 추가 출처"""
    src = meta["source"]
    ok = (~src.isin(EXTRA_SOURCES) | src.isin(EXTRA[exp])).to_numpy().copy()
    if exp == "E0":
        ok &= meta["label"].isin(DANGER).to_numpy()
    return ok


def threshold_metrics(model, X, y, thr):
    proba = model.predict_proba(X)
    cls = list(model.classes_)
    dcols = [cls.index(c) for c in DANGER if c in cls]
    out = {}
    is_other = (y == "OTHER")
    if is_other.any() and dcols:
        alarm = proba[:, dcols].max(axis=1) >= thr
        out["other_false_alarm@thr"] = round(float(alarm[is_other].mean()), 3)
    for c in DANGER:
        m = (y == c)
        if m.any() and c in cls:
            out[f"{c}_recall@thr"] = round(float((proba[m, cls.index(c)] >= thr).mean()), 3)
    return out


def evaluate(exp, model, X, y, groups_desc, rep_dir, thr):
    pred = model.predict(X)
    labels = sorted(set(y) | set(model.classes_))
    present = sorted(set(y))  # 평가 데이터에 실제로 있는 클래스만 macro F1 에 포함
    rep = classification_report(y, pred, labels=present, output_dict=True, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=[f"true_{l}" for l in labels],
                      columns=[f"pred_{l}" for l in labels])
    cm.to_csv(rep_dir / f"val_confusion_{exp}.csv", encoding="utf-8-sig")
    row = {"exp": exp, "eval": groups_desc, "n_eval": len(y), "macro_f1": round(rep["macro avg"]["f1-score"], 3)}
    for c in present:
        if c in rep:
            row[f"{c}_recall"] = round(rep[c]["recall"], 3)
    row.update(threshold_metrics(model, X, y, thr))
    print(f"\n=== {exp} ({groups_desc}) ===\n{cm}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="E1", choices=list(EXTRA))
    args = ap.parse_args()

    cfg = load_json("train.json")
    thr = cfg["danger_threshold"]
    d = np.load(DATA_DIR / "windows" / "windows.npz", allow_pickle=True)
    X, y = d["X"], d["y"].astype(str)
    meta = pd.read_csv(DATA_DIR / "windows" / "windows_meta.csv")
    rep_dir = ensure_dir(REPORT_DIR / "day06")
    present = set(meta["source"])
    exps = [e for e, need in EXTRA.items() if need <= present]
    if args.save not in exps:
        raise SystemExit(f"{args.save} 에 필요한 데이터가 없습니다 ({EXTRA[args.save]}). 가능한 실험: {exps}")

    is_train = (meta["split"] == "train").to_numpy()
    is_val = (meta["split"] == "val").to_numpy()
    rows, models = [], {}
    for exp in exps:
        m_tr = is_train & exp_mask(meta, exp)
        model = make_rf(cfg).fit(X[m_tr], y[m_tr])
        models[exp] = (model, m_tr)
        if is_val.any():
            # 같은 Val 로 비교 (외부 데이터는 Val 에 없음). E0 은 위험 창만이 아니라 OTHER 창까지 포함해 오경보를 본다
            rows.append(evaluate(exp, model, X[is_val], y[is_val], "Val", rep_dir, thr))
        else:
            # Val 이 없으면 Train(외부 제외) 안에서 영상 단위 GroupKFold — 참고용, 같은 사람이라 낙관적임
            base = is_train & (~meta["source"].isin(EXTRA_SOURCES)).to_numpy()
            groups = meta["video"].to_numpy()
            idx = np.where(base)[0]
            n_split = min(5, len(set(groups[idx])))
            for tr_i, te_i in GroupKFold(n_splits=n_split).split(idx, groups=groups[idx]):
                tr_idx, te_idx = idx[tr_i], idx[te_i]
                tr_idx = tr_idx[exp_mask(meta.iloc[tr_idx], exp)]
                # 외부 데이터는 항상 학습 쪽에만
                if EXTRA[exp]:
                    # 추가 데이터 중 '평가 fold 와 같은 Take 의 VICON 2D' 는 빼서 누수를 막는다
                    test_takes = set(meta["video"].iloc[te_idx])
                    take_of = meta["video"].str.slice(0, 9)
                    extra = meta["source"].isin(EXTRA[exp]) & ~take_of.isin(test_takes)
                    tr_idx = np.concatenate([tr_idx, np.where(extra)[0]])
                fm = make_rf(cfg).fit(X[tr_idx], y[tr_idx])
                rows_fold = evaluate(f"{exp}_fold", fm, X[te_idx], y[te_idx], "GroupKFold", rep_dir, thr)
                rows_fold["exp"] = exp
                rows.append(rows_fold)
    res = pd.DataFrame(rows)
    if "eval" in res and (res["eval"] == "GroupKFold").any():
        res = res.groupby(["exp", "eval"]).mean(numeric_only=True).round(3).reset_index()
        print("\n[주의] Val 이 비어 있어 GroupKFold(영상 단위, 같은 사람) 결과입니다. 실제보다 좋게 나옵니다.")
    res.to_csv(rep_dir / "val_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"\n[비교]\n{res.to_string(index=False)}")

    # 서비스용 모델 3개 저장 (수업과 같은 파일 이름)
    model, m_tr = models[args.save]
    ensure_dir(MODEL_DIR)
    joblib.dump(model, MODEL_DIR / "rf_action_baseline.joblib")
    names = sequence_feature_names(cfg["sequence_steps"])
    (MODEL_DIR / "rf_feature_columns.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    info = {
        "model_version": f"bareunpum-{args.save}",
        "experiment": args.save,
        "classes": [str(c) for c in model.classes_],
        "feature_count": len(names),
        "sequence_steps": cfg["sequence_steps"],
        "frame_feature_count": len(names) // cfg["sequence_steps"],
        "window_seconds": cfg["window_seconds"],
        "window_stride_seconds": cfg["window_stride_seconds"],
        "min_valid_frames": cfg["min_valid_frames"],
        "danger_threshold": thr,
        "repeat_windows": cfg["repeat_windows"],
        "train_windows": int(m_tr.sum()),
        "train_performers": sorted(meta.loc[m_tr, "performer_id"].dropna().astype(str).unique().tolist()),
        "train_sources": sorted(meta.loc[m_tr, "source"].unique().tolist()),
        "n_estimators": cfg["n_estimators"], "max_depth": cfg["max_depth"],
        "class_weight": cfg.get("class_weight"), "random_state": cfg["random_state"],
        "sklearn_version": sklearn.__version__,
        "note": "predict_proba 는 보정된 위험 확률이 아님. 임계값·반복 횟수는 Val 기준 초기값.",
    }
    (MODEL_DIR / "rf_training_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {MODEL_DIR}/ rf_action_baseline.joblib · rf_feature_columns.json · rf_training_info.json  ({args.save}, classes={info['classes']})")


if __name__ == "__main__":
    main()
