from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor

from run_pipeline import ExperimentConfig, build_labeled_dataset, load_dataset, make_xy


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_state_summary(labeled_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_rows = (
        labeled_dataset.groupby("stateKey", observed=False)
        .agg(
            stateCbrBin=("stateCbrBin", "first"),
            stateNeighborBin=("stateNeighborBin", "first"),
            stateHopBin=("stateHopBin", "first"),
            stateDirectBin=("stateDirectBin", "first"),
            bestLow=("bestLow", "first"),
            bestHigh=("bestHigh", "first"),
            bestPairSampleCount=("bestPairSampleCount", "first"),
            bestScore=("bestScore", "first"),
            rows=("stateKey", "size"),
        )
        .reset_index()
    )

    pair_score = (
        labeled_dataset.groupby(["stateKey", "lowInt", "highInt"], observed=False)
        .agg(
            scoreMean=("offlineScore", "mean"),
            sampleCount=("offlineScore", "size"),
        )
        .reset_index()
        .sort_values(["stateKey", "scoreMean", "sampleCount"], ascending=[True, False, False])
    )
    pair_score["rank"] = pair_score.groupby("stateKey", observed=False).cumcount() + 1
    return state_rows, pair_score


def analyze_top2_gap(pair_score: pd.DataFrame) -> pd.DataFrame:
    top2 = pair_score[pair_score["rank"] <= 2].copy()
    top2_pivot = top2.pivot_table(
        index="stateKey",
        columns="rank",
        values=["scoreMean", "sampleCount", "lowInt", "highInt"],
        aggfunc="first",
    )
    top2_pivot.columns = [f"{name}_rank{rank}" for name, rank in top2_pivot.columns]
    top2_pivot = top2_pivot.reset_index()
    top2_pivot["score_gap_top1_top2"] = top2_pivot["scoreMean_rank1"] - top2_pivot["scoreMean_rank2"]
    top2_pivot["low_gap_top1_top2"] = np.abs(top2_pivot["lowInt_rank1"] - top2_pivot["lowInt_rank2"])
    top2_pivot["high_gap_top1_top2"] = np.abs(top2_pivot["highInt_rank1"] - top2_pivot["highInt_rank2"])
    return top2_pivot


def analyze_neighbor_smoothness(state_rows: pd.DataFrame) -> pd.DataFrame:
    rows = state_rows.copy().reset_index(drop=True)
    feature_matrix = rows[["stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin"]].to_numpy(dtype=float)
    target_matrix = rows[["bestLow", "bestHigh"]].to_numpy(dtype=float)

    neighbor_records = []
    for idx in range(len(rows)):
        diff = np.abs(feature_matrix - feature_matrix[idx])
        manhattan = diff.sum(axis=1)
        manhattan[idx] = np.inf
        nearest_idx = int(np.argmin(manhattan))
        neighbor_records.append(
            {
                "stateKey": rows.loc[idx, "stateKey"],
                "nearestStateKey": rows.loc[nearest_idx, "stateKey"],
                "nearestStateDistance": float(manhattan[nearest_idx]),
                "labelDistanceL1": float(np.abs(target_matrix[idx] - target_matrix[nearest_idx]).sum()),
                "lowDiff": float(np.abs(target_matrix[idx, 0] - target_matrix[nearest_idx, 0])),
                "highDiff": float(np.abs(target_matrix[idx, 1] - target_matrix[nearest_idx, 1])),
            }
        )
    return pd.DataFrame(neighbor_records)


def evaluate_state_level_knn(state_rows: pd.DataFrame) -> pd.DataFrame:
    x_state = state_rows[["stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin"]].astype(float)
    y_state = state_rows[["bestLow", "bestHigh"]].astype(float)
    groups = state_rows["stateKey"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(x_state, y_state, groups=groups))

    rows = []
    for k in [1, 3, 5, 9]:
        model = KNeighborsRegressor(n_neighbors=k, weights="distance")
        model.fit(x_state.iloc[train_idx], y_state.iloc[train_idx])
        pred = model.predict(x_state.iloc[test_idx])
        rows.append(
            {
                "model": f"state_knn_k{k}",
                "r2_low": float(r2_score(y_state.iloc[test_idx, 0], pred[:, 0])),
                "r2_high": float(r2_score(y_state.iloc[test_idx, 1], pred[:, 1])),
                "r2_uniform": float(r2_score(y_state.iloc[test_idx], pred, multioutput="uniform_average")),
            }
        )
    return pd.DataFrame(rows).sort_values("r2_uniform", ascending=False).reset_index(drop=True)


def main() -> None:
    config = ExperimentConfig(
        experiment_name="failure_analysis_reference",
        state_cbr_bin_size=5.0,
        state_neighbor_bin_size=5.0,
        min_pair_samples_per_state=10,
        include_run_pdr_in_label=False,
    )
    dataset = load_dataset()
    labeled_dataset = build_labeled_dataset(dataset, config)
    _, _, _ = make_xy(labeled_dataset)

    state_rows, pair_score = build_state_summary(labeled_dataset)
    top2_gap = analyze_top2_gap(pair_score)
    neighbor_smoothness = analyze_neighbor_smoothness(state_rows)
    knn_result = evaluate_state_level_knn(state_rows)

    merged = state_rows.merge(top2_gap, on="stateKey", how="left").merge(
        neighbor_smoothness,
        on="stateKey",
        how="left",
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "row_count",
                "unique_states",
                "median_bestPairSampleCount",
                "p25_bestPairSampleCount",
                "p75_bestPairSampleCount",
                "median_score_gap_top1_top2",
                "p25_score_gap_top1_top2",
                "median_nearest_state_distance",
                "median_neighbor_label_l1_distance",
                "share_score_gap_lt_0.5",
                "share_neighbor_label_l1_ge_4",
            ],
            "value": [
                len(labeled_dataset),
                labeled_dataset["stateKey"].nunique(),
                float(merged["bestPairSampleCount"].median()),
                float(merged["bestPairSampleCount"].quantile(0.25)),
                float(merged["bestPairSampleCount"].quantile(0.75)),
                float(merged["score_gap_top1_top2"].median()),
                float(merged["score_gap_top1_top2"].quantile(0.25)),
                float(merged["nearestStateDistance"].median()),
                float(merged["labelDistanceL1"].median()),
                float((merged["score_gap_top1_top2"] < 0.5).mean()),
                float((merged["labelDistanceL1"] >= 4.0).mean()),
            ],
        }
    )

    summary_path = OUTPUT_DIR / "failure_analysis_summary.csv"
    state_path = OUTPUT_DIR / "failure_analysis_state_rows.csv"
    knn_path = OUTPUT_DIR / "failure_analysis_state_knn.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    merged.to_csv(state_path, index=False, encoding="utf-8-sig")
    knn_result.to_csv(knn_path, index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print()
    print(knn_result.to_string(index=False))
    print()
    print(f"saved: {summary_path}")
    print(f"saved: {state_path}")
    print(f"saved: {knn_path}")


if __name__ == "__main__":
    main()
