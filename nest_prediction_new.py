#!/usr/bin/env python3
"""
NEST 2026 PREDICTION SYSTEM v6.1 (NaN-safe)
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("nest2026")


@dataclass
class Config:
    data_path: str = "all_merged_cleaned (new).json"
    target_year: int = 2026
    min_appearances: int = 2
    min_years_for_trend: int = 3
    recency_tau: float = 3.0
    rolling_windows: tuple = (2, 3)
    recent_start_year: int = 2020
    exclude_years: tuple = (2021,)
    cv_splits: int = 3
    calibration_method: str = "sigmoid"
    top_n: int = 20
    output_path: str = "nest2026_predictions_v6.json"


# ----------------------------------------------------------------------
# Data loading (same as before, omitted for brevity)
# ----------------------------------------------------------------------
# Keep this as a dict. If no normalization map is available, use an empty map.
TOPIC_NORM: dict[str, str] = {}

def load_and_clean(data_path: str) -> list[dict]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    try:
        raw = json.loads(text)
        if not isinstance(raw, list):
            raw = [raw]
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    records.extend(obj)
                else:
                    records.append(obj)
            except Exception:
                continue
        raw = records
    cleaned = []
    for q in raw:
        subject = str(q.get("subject", "")).strip()
        if subject == "General":
            continue
        chapter = str(q.get("chapter") or "").strip()
        topic_raw = str(q.get("topic") or chapter or subject).strip()
        topic_map = TOPIC_NORM if isinstance(TOPIC_NORM, dict) else {}
        topic = topic_map.get(topic_raw, topic_raw) or subject
        chapter_norm = topic_map.get(chapter, chapter) or subject
        year = q.get("year")
        if year is None:
            continue
        year = int(year)
        cleaned.append({
            "year": year,
            "subject": subject,
            "chapter": chapter_norm,
            "topic": topic,
            "shift": str(q.get("shift", "Shift 1")).strip(),
            "difficulty": str(q.get("difficulty", "unknown")).lower(),
        })
    log.info(f"Loaded {len(cleaned)} records (General removed).")
    return cleaned


# ----------------------------------------------------------------------
# Feature extraction (NaN-safe)
# ----------------------------------------------------------------------
def build_chapter_year_features(records: list[dict], cfg: Config) -> pd.DataFrame:
    rows = []
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        key = (r["subject"], r["chapter"])
        groups[key][r["year"]].append(r)

    # Precompute historical prior (2007-2019)
    prior_dict = {}
    for (subject, chapter), year_dict in groups.items():
        hist_years = [y for y in year_dict.keys() if y < cfg.recent_start_year and y not in cfg.exclude_years]
        if hist_years:
            prior = len(hist_years) / (cfg.recent_start_year - min(hist_years) + 1)
        else:
            prior = 0.0
        prior_dict[(subject, chapter)] = prior

    for (subject, chapter), year_dict in groups.items():
        present_years = sorted(year_dict.keys())
        if not present_years:
            continue
        for year in range(min(present_years), cfg.target_year + 1):
            if year in cfg.exclude_years:
                continue
            hist_years = [y for y in present_years if y < year]
            if len(hist_years) < cfg.min_appearances and year != cfg.target_year:
                continue

            counts = [len(year_dict[y]) for y in hist_years]

            # Recency-weighted mean
            if hist_years:
                weights = [np.exp(-(year - y) / cfg.recency_tau) for y in hist_years]
                wmean = np.average(counts, weights=weights)
            else:
                wmean = 0.0

            # Last 3 appearances
            last3 = sum(1 for y in hist_years if y >= year - 3)

            # Total appearances
            total_app = len(hist_years)

            # Trend slope and p-value (handle insufficient data)
            if len(counts) >= cfg.min_years_for_trend:
                x = np.arange(len(counts))
                slope, _, _, p_val, _ = stats.linregress(x, counts)
            else:
                slope, p_val = 0.0, 1.0

            # Years since last seen
            last_seen = hist_years[-1] if hist_years else year - 10
            years_since = year - last_seen

            # Regularity (fraction of years present)
            if hist_years:
                span = max(hist_years) - min(hist_years) + 1
                regularity = len(hist_years) / max(span, 1)
            else:
                regularity = 0.0

            # Topic diversity
            all_topics = set()
            for y in hist_years:
                for q in year_dict[y]:
                    all_topics.add(q.get("topic", ""))
            topic_diversity = len(all_topics) / max(1, total_app)

            # Historical prior
            historical_prior = prior_dict.get((subject, chapter), 0.0)

            # Rolling averages
            rolling_2 = np.mean([c for y, c in zip(hist_years, counts) if y >= year - 2]) if any(y >= year - 2 for y in hist_years) else 0.0
            rolling_3 = np.mean([c for y, c in zip(hist_years, counts) if y >= year - 3]) if any(y >= year - 3 for y in hist_years) else 0.0

            # Core chapter
            core_chapters = {
                "Physics": {"Mechanics", "Kinematics", "Thermodynamics", "Electrostatics", "Electromagnetism", "Optics", "Quantum Mechanics", "Nuclear Physics", "Waves", "Oscillations", "Fluid Mechanics", "Gravitation"},
                "Chemistry": {"Organic Chemistry", "Organic Reactions", "Chemical Bonding", "Chemical Kinetics", "Chemical Equilibrium", "Electrochemistry", "Thermodynamics", "Acids and Bases", "Atomic Structure", "Solid State"},
                "Biology": {"Genetics", "Molecular Biology", "Evolution", "Ecology", "Cell Biology", "Biochemistry", "Plant Physiology", "Human Physiology", "Immunology"},
                "Mathematics": {"Calculus", "Algebra", "Probability", "Trigonometry", "Geometry", "Number Theory", "Combinatorics", "Linear Algebra", "Vectors", "Coordinate Geometry", "Differential Equations"},
            }
            is_core = int(chapter in core_chapters.get(subject, set()))

            feat = {
                "weighted_mean": wmean,
                "last3_appearances": last3,
                "total_appearances": total_app,
                "trend_slope": slope,
                "trend_p": p_val,
                "years_since_last": years_since,
                "regularity": regularity,
                "topic_diversity": topic_diversity,
                "historical_prior": historical_prior,
                "is_core": is_core,
                "rolling_mean_2": rolling_2,
                "rolling_mean_3": rolling_3,
            }

            label = int(year in year_dict)
            rows.append({
                "subject": subject,
                "chapter": chapter,
                "year": year,
                "label": label,
                **feat,
            })

    df = pd.DataFrame(rows)
    # Replace any remaining NaN/Inf with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    log.info(f"Built chapter‑year features: {len(df)} rows, years {df['year'].min()}–{df['year'].max()}")
    return df


# ----------------------------------------------------------------------
# Training with threshold tuning (NaN-safe)
# ----------------------------------------------------------------------
def train_subject_classifier_with_threshold(
    df: pd.DataFrame,
    subject: str,
    cfg: Config,
) -> tuple[Any, RobustScaler, list[str], float]:
    subj_df = df[df["subject"] == subject].copy()
    if subj_df.empty:
        return None, None, [], 0.5

    train_df = subj_df[(subj_df["year"] >= cfg.recent_start_year) & (subj_df["year"] != cfg.target_year)].copy()
    if train_df.empty:
        return None, None, [], 0.5

    feature_cols = [c for c in train_df.columns if c not in ["subject", "chapter", "year", "label"]]
    X = train_df[feature_cols].values.astype(float)
    y = train_df["label"].values.astype(int)

    # Replace any NaN that might have slipped
    X = np.nan_to_num(X, nan=0.0)

    if len(np.unique(y)) < 2:
        log.warning(f"Subject {subject}: only one class, skipping.")
        return None, None, [], 0.5

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))

    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight=class_weight_dict, random_state=42)
    lr = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=1000)

    voting = VotingClassifier(estimators=[("gb", gb), ("rf", rf), ("lr", lr)], voting="soft", weights=[1, 1, 1])

    # Calibration
    n_splits = min(cfg.cv_splits, max(2, len(X_scaled) // 10))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    calibrated = CalibratedClassifierCV(voting, method=cfg.calibration_method, cv=tscv if tscv.n_splits > 1 else 3)
    calibrated.fit(X_scaled, y)

    # Find best threshold
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.linspace(0.1, 0.9, 17)
    for th in thresholds:
        fold_f1 = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            y_tr, y_val = y[train_idx], y[val_idx]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue
            temp = clone(voting)
            temp.fit(X_tr, y_tr)
            y_prob = temp.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= th).astype(int)
            fold_f1.append(f1_score(y_val, y_pred, zero_division=0))
        if not fold_f1:
            continue
        mean_f1 = np.mean(fold_f1)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_threshold = th

    log.info(f"Subject {subject}: best threshold = {best_threshold:.3f} (F1={best_f1:.3f})")
    return calibrated, scaler, feature_cols, best_threshold


# ----------------------------------------------------------------------
# Validation (skip 2021, NaN-safe)
# ----------------------------------------------------------------------
def validate_recent_years(df: pd.DataFrame, cfg: Config) -> dict:
    years = sorted(df["year"].unique())
    recent_years = [y for y in years if y >= cfg.recent_start_year and y < cfg.target_year and y not in cfg.exclude_years]
    metrics = defaultdict(list)

    for test_year in recent_years:
        train_years = [y for y in years if y < test_year and y not in cfg.exclude_years]
        train_df = df[df["year"].isin(train_years)]
        test_df = df[df["year"] == test_year]

        models_subj = {}
        scalers_subj = {}
        thresholds_subj = {}
        feature_cols = [c for c in train_df.columns if c not in ["subject", "chapter", "year", "label"]]

        for subj in ["Physics", "Chemistry", "Biology", "Mathematics"]:
            subj_train = train_df[train_df["subject"] == subj]
            if len(subj_train) < 10 or subj_train["label"].sum() < 2:
                continue
            X_tr = subj_train[feature_cols].values.astype(float)
            y_tr = subj_train["label"].values.astype(int)
            X_tr = np.nan_to_num(X_tr, nan=0.0)

            scaler = RobustScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_tr_scaled = np.nan_to_num(X_tr_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            classes = np.unique(y_tr)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
            class_weight_dict = dict(zip(classes, weights))

            gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight=class_weight_dict, random_state=42)
            lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
            voting = VotingClassifier(estimators=[("gb", gb), ("rf", rf), ("lr", lr)], voting="soft")
            voting.fit(X_tr_scaled, y_tr)
            models_subj[subj] = voting
            scalers_subj[subj] = scaler

            # Find threshold for this subject
            best_th = 0.5
            best_f1 = 0.0
            n_splits = min(3, max(2, len(X_tr_scaled) // 10))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for th in np.linspace(0.1, 0.9, 17):
                fold_f1 = []
                for train_idx, val_idx in tscv.split(X_tr_scaled):
                    X_tr_cv, X_val_cv = X_tr_scaled[train_idx], X_tr_scaled[val_idx]
                    X_tr_cv = np.nan_to_num(X_tr_cv, nan=0.0, posinf=0.0, neginf=0.0)
                    X_val_cv = np.nan_to_num(X_val_cv, nan=0.0, posinf=0.0, neginf=0.0)
                    y_tr_cv, y_val_cv = y_tr[train_idx], y_tr[val_idx]
                    if len(np.unique(y_tr_cv)) < 2 or len(np.unique(y_val_cv)) < 2:
                        continue
                    m = clone(voting)
                    m.fit(X_tr_cv, y_tr_cv)
                    y_prob = m.predict_proba(X_val_cv)[:, 1]
                    y_pred = (y_prob >= th).astype(int)
                    fold_f1.append(f1_score(y_val_cv, y_pred, zero_division=0))
                if not fold_f1:
                    continue
                if np.mean(fold_f1) > best_f1:
                    best_f1 = np.mean(fold_f1)
                    best_th = th
            thresholds_subj[subj] = best_th

        # Predict on test year
        all_true, all_pred = [], []
        for subj in models_subj:
            subj_test = test_df[test_df["subject"] == subj]
            if subj_test.empty:
                continue
            X_te = subj_test[feature_cols].values.astype(float)
            y_true = subj_test["label"].values.astype(int)
            X_te = np.nan_to_num(X_te, nan=0.0)
            X_te_scaled = scalers_subj[subj].transform(X_te)
            X_te_scaled = np.nan_to_num(X_te_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            y_prob = models_subj[subj].predict_proba(X_te_scaled)[:, 1]
            th = thresholds_subj.get(subj, 0.5)
            y_pred = (y_prob >= th).astype(int)
            all_true.extend(y_true)
            all_pred.extend(y_pred)

        if all_true:
            acc = (np.array(all_true) == np.array(all_pred)).mean()
            prec = precision_score(all_true, all_pred, zero_division=0)
            rec = recall_score(all_true, all_pred, zero_division=0)
            f1 = f1_score(all_true, all_pred, zero_division=0)
            try:
                auc = roc_auc_score(all_true, y_prob) if len(np.unique(all_true)) > 1 else 0.5
            except:
                auc = 0.5
            metrics["year"].append(test_year)
            metrics["accuracy"].append(acc)
            metrics["precision"].append(prec)
            metrics["recall"].append(rec)
            metrics["f1"].append(f1)
            metrics["auc"].append(auc)
            log.info(f"Validation {test_year}: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}, auc={auc:.3f}")

    if metrics["year"]:
        summary = {
            "mean_accuracy": round(np.mean(metrics["accuracy"]), 4),
            "mean_precision": round(np.mean(metrics["precision"]), 4),
            "mean_recall": round(np.mean(metrics["recall"]), 4),
            "mean_f1": round(np.mean(metrics["f1"]), 4),
            "mean_auc": round(np.mean(metrics["auc"]), 4),
            "per_year": [
                {"year": y, "accuracy": a, "precision": p, "recall": r, "f1": f, "auc": auc}
                for y, a, p, r, f, auc in zip(metrics["year"], metrics["accuracy"],
                                               metrics["precision"], metrics["recall"],
                                               metrics["f1"], metrics["auc"])
            ]
        }
    else:
        summary = {}
    return summary


# ----------------------------------------------------------------------
# Prediction for 2026
# ----------------------------------------------------------------------
def predict_2026(
    df: pd.DataFrame,
    models: dict,
    scalers: dict,
    thresholds: dict,
    feature_cols: list,
    cfg: Config,
) -> dict:
    predictions = {}
    for subject in ["Physics", "Chemistry", "Biology", "Mathematics"]:
        if subject not in models or models[subject] is None:
            predictions[subject] = []
            continue

        subj_df = df[df["subject"] == subject]
        chapters = subj_df["chapter"].unique()
        pred_rows = []
        for ch in chapters:
            hist = subj_df[(subj_df["chapter"] == ch) & (subj_df["year"] < cfg.target_year)]
            if hist.empty:
                prob = 0.05
            else:
                # Compute features for target year using the same logic as in build_chapter_year_features
                years = hist["year"].values
                counts = hist["label"].values
                weights = np.exp(-(cfg.target_year - years) / cfg.recency_tau)
                wmean = np.average(counts, weights=weights) if weights.sum() > 0 else 0.0
                last3 = hist[hist["year"] >= cfg.target_year - 3]["label"].sum()
                total_app = len(hist)
                if len(counts) >= cfg.min_years_for_trend:
                    x = np.arange(len(counts))
                    slope, _, _, p_val, _ = stats.linregress(x, counts)
                else:
                    slope, p_val = 0.0, 1.0
                years_since = cfg.target_year - (years[-1] if len(years) > 0 else cfg.target_year - 10)
                if len(years) > 1:
                    span = years[-1] - years[0] + 1
                    regularity = len(years) / max(span, 1)
                else:
                    regularity = 0.0
                all_topics = set(hist["topic"].dropna().values)
                topic_diversity = len(all_topics) / max(1, total_app)
                prior = hist["historical_prior"].iloc[0] if "historical_prior" in hist.columns else 0.0
                rolling_2 = hist[hist["year"] >= cfg.target_year - 2]["label"].mean() if (hist["year"] >= cfg.target_year - 2).any() else 0.0
                rolling_3 = hist[hist["year"] >= cfg.target_year - 3]["label"].mean() if (hist["year"] >= cfg.target_year - 3).any() else 0.0
                core_ch = int(ch in {
                    "Physics": {"Mechanics", "Kinematics", "Thermodynamics", "Electrostatics", "Electromagnetism", "Optics", "Quantum Mechanics", "Nuclear Physics", "Waves", "Oscillations", "Fluid Mechanics", "Gravitation"},
                    "Chemistry": {"Organic Chemistry", "Organic Reactions", "Chemical Bonding", "Chemical Kinetics", "Chemical Equilibrium", "Electrochemistry", "Thermodynamics", "Acids and Bases", "Atomic Structure", "Solid State"},
                    "Biology": {"Genetics", "Molecular Biology", "Evolution", "Ecology", "Cell Biology", "Biochemistry", "Plant Physiology", "Human Physiology", "Immunology"},
                    "Mathematics": {"Calculus", "Algebra", "Probability", "Trigonometry", "Geometry", "Number Theory", "Combinatorics", "Linear Algebra", "Vectors", "Coordinate Geometry", "Differential Equations"},
                }.get(subject, set()))

                feat = np.array([[wmean, last3, total_app, slope, p_val, years_since, regularity,
                                  topic_diversity, prior, core_ch, rolling_2, rolling_3]])
                # Ensure same order as feature_cols
                feat_df = pd.DataFrame(feat, columns=feature_cols)
                # Replace any NaN
                feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)
                feat_scaled = scalers[subject].transform(feat_df)
                feat_scaled = np.nan_to_num(feat_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                prob = models[subject].predict_proba(feat_scaled)[0, 1]
                prob = float(np.clip(prob, 0.01, 0.99))

            pred_rows.append((ch, prob))

        threshold = thresholds.get(subject, 0.5)
        filtered = [(ch, prob) for ch, prob in pred_rows if prob >= threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        final = filtered[:cfg.top_n]
        predictions[subject] = [{"chapter": ch, "probability": round(prob, 4)} for ch, prob in final]
        log.info(f"{subject}: selected {len(final)} chapters (threshold={threshold:.3f})")
    return predictions


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    cfg = Config()
    log.info("NEST 2026 Prediction System v6.1 (NaN-safe)")
    log.info(f"Target year: {cfg.target_year}")

    records = load_and_clean(cfg.data_path)
    df = build_chapter_year_features(records, cfg)
    df = df[~df["year"].isin(cfg.exclude_years)]  # remove 2021
    log.info(f"After removing excluded years: {len(df)} rows")

    val_results = validate_recent_years(df, cfg)
    log.info(f"Validation summary: {val_results}")

    # Train final models
    models = {}
    scalers = {}
    thresholds = {}
    feature_cols = [c for c in df.columns if c not in ["subject", "chapter", "year", "label"]]

    for subj in ["Physics", "Chemistry", "Biology", "Mathematics"]:
        clf, scaler, cols, th = train_subject_classifier_with_threshold(df, subj, cfg)
        models[subj] = clf
        scalers[subj] = scaler
        thresholds[subj] = th
        if clf is not None:
            log.info(f"Trained {subj} classifier on {len(df[df['subject']==subj])} rows, threshold={th:.3f}")

    predictions = predict_2026(df, models, scalers, thresholds, feature_cols, cfg)

    output = {
        "meta": {
            "version": "6.1",
            "generated_at": datetime.now().isoformat(),
            "target_year": cfg.target_year,
            "config": asdict(cfg),
            "validation": val_results,
            "note": "Excluded 2021. NaN-safe."
        },
        "predictions": predictions
    }

    out_path = Path(cfg.output_path)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info(f"Saved predictions to {out_path}")

    print("\n" + "="*80)
    print(f"NEST 2026 PREDICTIONS (Top chapters per subject)")
    print("="*80)
    for subj, chap_list in predictions.items():
        print(f"\n{subj.upper()}:")
        for i, item in enumerate(chap_list[:15], 1):
            print(f"  {i:2}. {item['chapter']:<35} (prob {item['probability']:.3f})")
    print("="*80)


if __name__ == "__main__":
    main()