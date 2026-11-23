"""6일차 마지막 — 저장된 모델(models/)로 Test 를 '한 번만' 평가.

실행:  uv run python scripts/d06b_test_eval.py
입력:  models/rf_action_baseline.joblib, models/rf_training_info.json, data/windows/windows.npz·windows_meta.csv
결과:  reports/day06/test_metrics.csv, test_confusion.csv, test_per_video.csv
원칙:  임계값·실험 선택은 Val 로 끝낸 뒤 실행한다. Test 결과를 보고 다시 고치면 Test 가 Val 이 된다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from scripts.d06_train_rf import threshold_metrics
from src.common import DATA_DIR, MODEL_DIR, REPORT_DIR, ensure_dir


def main():
    model = joblib.load(MODEL_DIR / "rf_action_baseline.joblib")
    info = json.loads((MODEL_DIR / "rf_training_info.json").read_text(encoding="utf-8"))
    thr = info["danger_threshold"]
    d = np.load(DATA_DIR / "windows" / "windows.npz", allow_pickle=True)
    meta = pd.read_csv(DATA_DIR / "windows" / "windows_meta.csv")
    te = (meta["split"] == "test").to_numpy()
    X, y = d["X"][te], d["y"].astype(str)[te]
    if len(y) == 0:
        raise SystemExit("Test 창이 없습니다. split.json 을 확인하세요.")

    proba = model.predict_proba(X)
    cls = list(model.classes_)
    pred = np.array(cls)[proba.argmax(1)]
    rep = ensure_dir(REPORT_DIR / "day06")

    m = {"model_version": info.get("model_version"), "threshold": thr, "n_test": len(y),
         "argmax_acc": round(float((pred == y).mean()), 3)}
    m.update(threshold_metrics(model, X, y, thr))
    pd.DataFrame([m]).to_csv(rep / "test_metrics.csv", index=False, encoding="utf-8-sig")

    conf = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred"))
    conf.to_csv(rep / "test_confusion.csv", encoding="utf-8-sig")

    per = meta[te].copy()
    for c in cls:
        per[f"p_{c}"] = proba[:, cls.index(c)]
    danger = [c for c in cls if c != "OTHER"]
    per["alarm"] = per[[f"p_{c}" for c in danger]].max(axis=1) >= thr
    pv = per.groupby(["video", "label"]).agg(windows=("alarm", "size"), alarm_windows=("alarm", "sum"),
                                             p_shake_mean=("p_shake", "mean"), p_shake_max=("p_shake", "max")).round(2)
    pv.to_csv(rep / "test_per_video.csv", encoding="utf-8-sig")

    print(f"=== Test ({info.get('model_version')}, thr {thr}) ===")
    print(conf.to_string(), "\n")
    print(pd.DataFrame([m]).to_string(index=False), "\n")
    print(pv.to_string())


if __name__ == "__main__":
    main()
