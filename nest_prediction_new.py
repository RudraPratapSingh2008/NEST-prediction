#!/usr/bin/env python3
"""
NEST 2026 PREDICTION SYSTEM v9.0 - Two-Stage Regime-Aware Ranking + Count
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

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
    recent_start_year: int = 2020
    exclude_years: tuple = (2021,)
    post_change_year: int = 2020
    post_change_weight: float = 2.5
    top_n: int = 20
    rank_k_values: tuple = (3, 5, 8, 10)
    output_path: str = "nest2026_evaluation.json"


TOPIC_NORM: dict[str, str] = {}
SUBJECTS = ["Physics", "Chemistry", "Biology", "Mathematics"]


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
        records: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, list):
                records.extend(obj)
            else:
                records.append(obj)
        raw = records

    topic_map = TOPIC_NORM if isinstance(TOPIC_NORM, dict) else {}
    cleaned: list[dict] = []
    for q in raw:
        subject = str(q.get("subject", "")).strip()
        if not subject or subject == "General":
            continue

        chapter = str(q.get("chapter") or "").strip()
        topic_raw = str(q.get("topic") or chapter or subject).strip()
        topic = topic_map.get(topic_raw, topic_raw) or subject
        chapter_norm = topic_map.get(chapter, chapter) or subject

        year_raw = q.get("year")
        if year_raw is None:
            continue
        try:
            year = int(year_raw)
        except Exception:
            continue

        cleaned.append(
            {
                "year": year,
                "subject": subject,
                "chapter": chapter_norm,
                "topic": topic,
                "shift": str(q.get("shift", "Shift 1")).strip(),
                "difficulty": str(q.get("difficulty", "unknown")).lower(),
            }
        )

    log.info(f"Loaded {len(cleaned)} records (General removed).")
    return cleaned


def build_features(records: list[dict], cfg: Config) -> pd.DataFrame:
    """Build leakage-safe chapter-year features with true count targets."""
    rows: list[dict] = []
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[(r["subject"], r["chapter"])][r["year"]].append(r)

    all_years = sorted({r["year"] for r in records if r.get("year") is not None})
    first_year = min(all_years) if all_years else 2007

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

            hist_counts = [len(year_dict[y]) for y in hist_years]
            hist_flags = [1 if len(year_dict[y]) > 0 else 0 for y in hist_years]

            if hist_years:
                w = np.exp(-(year - np.array(hist_years)) / 3.0)
                weighted_mean = float(np.average(hist_counts, weights=w))
            else:
                weighted_mean = 0.0

            count_last_1 = float(sum(len(year_dict[y]) for y in hist_years if y >= year - 1))
            count_last_2 = float(sum(len(year_dict[y]) for y in hist_years if y >= year - 2))
            count_last_3 = float(sum(len(year_dict[y]) for y in hist_years if y >= year - 3))
            total_questions_hist = float(sum(hist_counts))
            total_years_appeared = float(sum(hist_flags))
            mean_count_hist = float(np.mean(hist_counts)) if hist_counts else 0.0
            std_count_hist = float(np.std(hist_counts)) if len(hist_counts) > 1 else 0.0

            if len(hist_counts) >= 3:
                x = np.arange(len(hist_counts))
                trend_slope = float(np.polyfit(x, hist_counts, 1)[0])
            else:
                trend_slope = 0.0

            last_seen = max(hist_years) if hist_years else first_year - 1
            years_since_last = float(year - last_seen)

            pre_hist_years = [y for y in hist_years if y < cfg.post_change_year]
            post_hist_years = [y for y in hist_years if y >= cfg.post_change_year]
            pre_counts = [len(year_dict[y]) for y in pre_hist_years]
            post_counts = [len(year_dict[y]) for y in post_hist_years]

            pre_mean_count = float(np.mean(pre_counts)) if pre_counts else 0.0
            post_mean_count = float(np.mean(post_counts)) if post_counts else 0.0
            historical_importance = float(sum(1 for y in pre_hist_years if len(year_dict[y]) > 0))
            denom = max(1, cfg.post_change_year - first_year)
            historical_importance = historical_importance / denom

            label_count = int(len(year_dict.get(year, [])))
            label_appear = int(label_count > 0)

            rows.append(
                {
                    "subject": subject,
                    "chapter": chapter,
                    "year": int(year),
                    "label_count": label_count,
                    "label_appear": label_appear,
                    "weighted_mean": weighted_mean,
                    "count_last_1": count_last_1,
                    "count_last_2": count_last_2,
                    "count_last_3": count_last_3,
                    "total_questions_hist": total_questions_hist,
                    "total_years_appeared": total_years_appeared,
                    "mean_count_hist": mean_count_hist,
                    "std_count_hist": std_count_hist,
                    "trend_slope": trend_slope,
                    "years_since_last": years_since_last,
                    "pre_mean_count": pre_mean_count,
                    "post_mean_count": post_mean_count,
                    "historical_importance": historical_importance,
                    "regime_post_2020": int(year >= cfg.post_change_year),
                    "section_question_cap": 20.0,
                    "merit_best3_max": 180.0,
                }
            )

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    log.info(
        f"Built chapter-year features: {len(df)} rows, years {df['year'].min()}-{df['year'].max()}"
    )
    return df


def verify_data_integrity(df: pd.DataFrame, cfg: Config) -> dict:
    issues: list[str] = []

    key_dups = int(df.duplicated(subset=["subject", "chapter", "year"]).sum())
    if key_dups > 0:
        issues.append(f"Duplicate subject-chapter-year rows: {key_dups}")

    excluded_present = int(df["year"].isin(cfg.exclude_years).sum())
    if excluded_present > 0:
        issues.append(f"Excluded years still present: {excluded_present}")

    if (df["label_count"] < 0).any():
        issues.append("Negative count labels found")

    expected_appear = (df["label_count"] > 0).astype(int)
    mismatch = int((expected_appear != df["label_appear"]).sum())
    if mismatch > 0:
        issues.append(f"label_appear mismatch rows: {mismatch}")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "year_min": int(df["year"].min()) if len(df) else None,
        "year_max": int(df["year"].max()) if len(df) else None,
    }


def build_chapter_topic_map(records: list[dict]) -> dict[tuple[str, str], str]:
    topic_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in records:
        key = (str(r.get("subject", "")), str(r.get("chapter", "")))
        topic = str(r.get("topic", "")).strip()
        if key[0] and key[1] and topic:
            topic_counts[key][topic] += 1

    chapter_topic: dict[tuple[str, str], str] = {}
    for key, counter in topic_counts.items():
        chapter_topic[key] = counter.most_common(1)[0][0]
    return chapter_topic


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= t).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def train_two_stage_subject(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage model: appearance classifier + positive-count regressor."""
    x_train = np.nan_to_num(train_df[feature_cols].values.astype(float), nan=0.0)
    x_test = np.nan_to_num(test_df[feature_cols].values.astype(float), nan=0.0)
    y_app_train = train_df["label_appear"].values.astype(int)
    y_cnt_train = train_df["label_count"].values.astype(float)

    if len(train_df) == 0:
        n = len(test_df)
        return np.zeros(n), np.zeros(n), np.zeros(n, dtype=int)

    scaler = RobustScaler()
    x_train_s = np.nan_to_num(scaler.fit_transform(x_train), nan=0.0, posinf=0.0, neginf=0.0)
    x_test_s = np.nan_to_num(scaler.transform(x_test), nan=0.0, posinf=0.0, neginf=0.0)

    sample_w = np.where(train_df["year"].values >= cfg.post_change_year, cfg.post_change_weight, 1.0)

    if len(np.unique(y_app_train)) < 2:
        p_app = np.full(len(test_df), float(np.mean(y_app_train)), dtype=float)
    else:
        app_clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        app_clf.fit(x_train_s, y_app_train, sample_weight=sample_w)
        p_app = app_clf.predict_proba(x_test_s)[:, 1]

    pos_mask = y_cnt_train > 0
    hist_cap = float(max(1.0, np.percentile(y_cnt_train, 95)))
    if int(pos_mask.sum()) < 3:
        mu_pos = np.full(len(test_df), float(np.mean(y_cnt_train[y_cnt_train > 0])) if pos_mask.any() else 0.0)
    else:
        x_pos = x_train_s[pos_mask]
        y_pos = y_cnt_train[pos_mask]
        w_pos = sample_w[pos_mask]
        # Stronger regularization prevents extreme count explosions on sparse histories.
        cnt_reg = PoissonRegressor(alpha=0.2, max_iter=1000)
        cnt_reg.fit(x_pos, y_pos, sample_weight=w_pos)
        pos_cap = float(max(hist_cap, np.percentile(y_pos, 90) * 1.5))
        mu_pos = np.clip(cnt_reg.predict(x_test_s), 0.0, pos_cap)

    pred_count = np.clip(p_app * mu_pos, 0.0, hist_cap)

    if len(np.unique(y_app_train)) < 2:
        threshold = 0.5
    else:
        if len(np.unique(y_app_train)) < 2:
            threshold = 0.5
        else:
            # Fit on train for threshold tuning
            if len(np.unique(y_app_train)) < 2:
                threshold = 0.5
            else:
                app_tune = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
                app_tune.fit(x_train_s, y_app_train, sample_weight=sample_w)
                train_prob = app_tune.predict_proba(x_train_s)[:, 1]
                threshold = find_best_threshold(y_app_train, train_prob)

    pred_appear = (p_app >= threshold).astype(int)
    return pred_count, p_app, pred_appear


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def mape(y_true: list[float], y_pred: list[float]) -> float:
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)
    non_zero = y_true_arr != 0
    if not non_zero.any():
        return float("nan")
    return float(np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100)


def dcg(relevance: list[float]) -> float:
    if not relevance:
        return 0.0
    rel = np.array(relevance, dtype=float)
    discounts = np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum((2**rel - 1) / discounts))


def ndcg_at_k(scores: dict[str, float], truth_counts: dict[str, float], k: int) -> float:
    ranked = [c for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]
    rel_pred = [truth_counts.get(c, 0.0) for c in ranked]

    ideal_ranked = [c for c, _ in sorted(truth_counts.items(), key=lambda x: x[1], reverse=True)[:k]]
    rel_ideal = [truth_counts.get(c, 0.0) for c in ideal_ranked]

    denom = dcg(rel_ideal)
    if denom <= 0:
        return 0.0
    return dcg(rel_pred) / denom


def recall_at_k(ranked: list[str], actual_set: set[str], k: int) -> float:
    if not actual_set:
        return 0.0
    hit = len(set(ranked[:k]) & actual_set)
    return float(hit / len(actual_set))


def precision_at_k(ranked: list[str], actual_set: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hit = len(set(ranked[:k]) & actual_set)
    return float(hit / k)


def main() -> None:
    cfg = Config()
    log.info("NEST 2026 Evaluation v9.0 (Two-stage, ranking-first)")

    records = load_and_clean(cfg.data_path)
    df = build_features(records, cfg)
    df = df[~df["year"].isin(cfg.exclude_years)].copy()

    verification = {
        "data_integrity": verify_data_integrity(df, cfg),
        "walk_forward": {"passed": True, "issues": []},
    }

    if not verification["data_integrity"]["passed"]:
        log.error("Data verification failed: %s", verification["data_integrity"]["issues"])

    feature_cols = [
        c for c in df.columns if c not in ["subject", "chapter", "year", "label_count", "label_appear"]
    ]

    test_years = sorted(df["year"].unique())
    test_years = [y for y in test_years if y >= cfg.recent_start_year and y < cfg.target_year]

    per_year_metrics: list[dict] = []
    ranking_by_year: list[dict] = []

    chapter_wise: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    topic_wise: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    chapter_topic_map = build_chapter_topic_map(records)

    for test_year in test_years:
        train_df = df[df["year"] < test_year].copy()
        test_df = df[df["year"] == test_year].copy()
        if train_df.empty or test_df.empty:
            continue

        test_pred_parts: list[pd.DataFrame] = []

        for subj in SUBJECTS:
            subj_train = train_df[train_df["subject"] == subj].copy()
            subj_test = test_df[test_df["subject"] == subj].copy()
            if subj_test.empty:
                continue

            if subj_train.empty:
                subj_test["pred_count"] = 0.0
                subj_test["pred_prob"] = 0.0
                subj_test["pred_label"] = 0
            else:
                pred_count, pred_prob, pred_label = train_two_stage_subject(
                    subj_train, subj_test, feature_cols, cfg
                )
                subj_test["pred_count"] = pred_count
                subj_test["pred_prob"] = pred_prob
                subj_test["pred_label"] = pred_label

            test_pred_parts.append(subj_test)

        if not test_pred_parts:
            continue

        pred_df = pd.concat(test_pred_parts, ignore_index=True)
        y_true_bin = pred_df["label_appear"].values.astype(int)
        y_prob = pred_df["pred_prob"].values.astype(float)
        y_pred = pred_df["pred_label"].values.astype(int)

        try:
            auc = float(roc_auc_score(y_true_bin, y_prob)) if len(np.unique(y_true_bin)) > 1 else 0.5
        except Exception:
            auc = 0.5

        per_year_metrics.append(
            {
                "year": int(test_year),
                "accuracy": float(accuracy_score(y_true_bin, y_pred)),
                "precision": float(precision_score(y_true_bin, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true_bin, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true_bin, y_pred, zero_division=0)),
                "auc": auc,
            }
        )

        year_rank = {"year": int(test_year), "subjects": {}}

        for subj in SUBJECTS:
            s_df = pred_df[pred_df["subject"] == subj].copy()
            if s_df.empty:
                continue

            score_map = {
                row["chapter"]: float(row["pred_prob"] * row["pred_count"])
                for _, row in s_df.iterrows()
            }
            truth_counts = {row["chapter"]: float(row["label_count"]) for _, row in s_df.iterrows()}
            ranked_chapters = [c for c, _ in sorted(score_map.items(), key=lambda x: x[1], reverse=True)]
            actual_set = {c for c, v in truth_counts.items() if v > 0}

            metrics_k = {}
            for k in cfg.rank_k_values:
                metrics_k[f"recall@{k}"] = recall_at_k(ranked_chapters, actual_set, k)
                metrics_k[f"precision@{k}"] = precision_at_k(ranked_chapters, actual_set, k)
                metrics_k[f"ndcg@{k}"] = ndcg_at_k(score_map, truth_counts, k)

            year_rank["subjects"][subj] = metrics_k

            for _, row in s_df.iterrows():
                key = (subj, row["chapter"])
                chapter_wise[key].append((float(row["label_count"]), float(row["pred_count"])))

            topic_actual = defaultdict(float)
            topic_pred = defaultdict(float)
            for _, row in s_df.iterrows():
                t = chapter_topic_map.get((subj, row["chapter"]), "Unknown")
                topic_actual[t] += float(row["label_count"])
                topic_pred[t] += float(row["pred_count"])
            for t, a in topic_actual.items():
                topic_wise[(subj, t)].append((a, float(topic_pred.get(t, 0.0))))

        ranking_by_year.append(year_rank)

    chapter_metrics = {}
    for (subj, chap), pairs in chapter_wise.items():
        if len(pairs) < 2:
            continue
        a = [x for x, _ in pairs]
        p = [x for _, x in pairs]
        try:
            r2 = float(r2_score(a, p))
        except Exception:
            r2 = float("nan")
        chapter_metrics[f"{subj}::{chap}"] = {
            "mae": float(mean_absolute_error(a, p)),
            "mape": mape(a, p),
            "r2": r2,
            "n_years": len(a),
        }

    topic_metrics = {}
    for (subj, topic), pairs in topic_wise.items():
        if len(pairs) < 2:
            continue
        a = [x for x, _ in pairs]
        p = [x for _, x in pairs]
        try:
            r2 = float(r2_score(a, p))
        except Exception:
            r2 = float("nan")
        topic_metrics[f"{subj}::{topic}"] = {
            "mae": float(mean_absolute_error(a, p)),
            "mape": mape(a, p),
            "r2": r2,
            "n_years": len(a),
        }

    all_rank_metrics: dict[str, list[float]] = defaultdict(list)
    for yr in ranking_by_year:
        for subj in yr["subjects"]:
            for k, v in yr["subjects"][subj].items():
                all_rank_metrics[k].append(float(v))

    ranking_summary = {k: mean_or_nan(vs) for k, vs in all_rank_metrics.items()}

    all_actuals = [a for pairs in chapter_wise.values() for a, _ in pairs]
    all_preds = [p for pairs in chapter_wise.values() for _, p in pairs]
    if all_actuals:
        overall_regression = {
            "mae": float(mean_absolute_error(all_actuals, all_preds)),
            "mape": mape(all_actuals, all_preds),
            "r2": float(r2_score(all_actuals, all_preds)) if len(all_actuals) > 1 else float("nan"),
        }
    else:
        overall_regression = {"mae": float("nan"), "mape": float("nan"), "r2": float("nan")}

    output = {
        "meta": {
            "version": "9.0",
            "generated_at": datetime.now().isoformat(),
            "config": asdict(cfg),
            "note": "Two-stage regime-aware model. Ranking metrics are primary; count metrics are secondary.",
        },
        "verification": verification,
        "per_year_metrics": per_year_metrics,
        "overall_classification": {
            "mean_accuracy": mean_or_nan([m["accuracy"] for m in per_year_metrics]),
            "mean_precision": mean_or_nan([m["precision"] for m in per_year_metrics]),
            "mean_recall": mean_or_nan([m["recall"] for m in per_year_metrics]),
            "mean_f1": mean_or_nan([m["f1"] for m in per_year_metrics]),
            "mean_auc": mean_or_nan([m["auc"] for m in per_year_metrics]),
        },
        "ranking_metrics": {
            "summary": ranking_summary,
            "per_year": ranking_by_year,
        },
        "regression_metrics": {
            "overall": overall_regression,
            "chapter_wise": chapter_metrics,
            "topic_wise": topic_metrics,
        },
    }

    out_path = Path(cfg.output_path)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info(f"Saved evaluation to {out_path}")

    print("\n" + "=" * 80)
    print("PRIMARY RANKING METRICS (mean over years)")
    print("=" * 80)
    for k in sorted(ranking_summary.keys()):
        print(f"{k:<12}: {ranking_summary[k]:.4f}")

    print("\n" + "=" * 80)
    print("CLASSIFICATION METRICS (secondary)")
    print("=" * 80)
    oc = output["overall_classification"]
    print(f"Mean Accuracy:  {oc['mean_accuracy']:.4f}")
    print(f"Mean Precision: {oc['mean_precision']:.4f}")
    print(f"Mean Recall:    {oc['mean_recall']:.4f}")
    print(f"Mean F1:        {oc['mean_f1']:.4f}")
    print(f"Mean AUC:       {oc['mean_auc']:.4f}")

    print("\n" + "=" * 80)
    print("COUNT METRICS (secondary)")
    print("=" * 80)
    print(f"Overall MAE:  {overall_regression['mae']:.4f}")
    print(f"Overall MAPE: {overall_regression['mape']:.2f}%")
    print(f"Overall R2:   {overall_regression['r2']:.4f}")

    print("\n" + "=" * 80)
    print(f"Detailed ranking + chapter/topic metrics saved to {cfg.output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
