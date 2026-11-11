from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor

from run_pipeline import ExperimentConfig, load_dataset
from experiment_soft_labels import build_pair_score


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_smoothed_targets(
    pair_score: pd.DataFrame,
    bandwidth: float,
    temperature: float,
) -> pd.DataFrame:
    states = (
        pair_score[["stateKey", "stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    state_features = states[["stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin"]].to_numpy(dtype=float)

    pair_rows = pair_score.copy().reset_index(drop=True)
    pair_features = pair_rows[["stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin"]].to_numpy(dtype=float)

    score_shift = pair_rows["scoreMean"] - pair_rows.groupby("stateKey", observed=False)["scoreMean"].transform("max")
    pair_rows["pairWeightLocal"] = np.exp(score_shift / temperature) * pair_rows["sampleCount"].clip(lower=1)

    smoothed_records = []
    for idx, state_row in states.iterrows():
        target = state_features[idx]
        diff = np.abs(pair_features - target)
        dist = diff[:, 0] / 5.0 + diff[:, 1] / 5.0 + diff[:, 2] + 2.0 * diff[:, 3]
        kernel = np.exp(-dist / bandwidth)
        total_weight = kernel * pair_rows["pairWeightLocal"].to_numpy(dtype=float)
        weight_sum = float(total_weight.sum())
        if weight_sum <= 0:
            continue

        smoothed_low = float(np.sum(total_weight * pair_rows["lowInt"].to_numpy(dtype=float)) / weight_sum)
        smoothed_high = float(np.sum(total_weight * pair_rows["highInt"].to_numpy(dtype=float)) / weight_sum)
        smoothed_records.append(
            {
                "stateKey": state_row["stateKey"],
                "stateCbrBin": float(state_row["stateCbrBin"]),
                "stateNeighborBin": float(state_row["stateNeighborBin"]),
                "stateHopBin": float(state_row["stateHopBin"]),
                "stateDirectBin": float(state_row["stateDirectBin"]),
                "targetLow": smoothed_low,
                "targetHigh": smoothed_high,
                "supportWeight": weight_sum,
            }
        )
    return pd.DataFrame(smoothed_records)


def evaluate_targets(state_targets: pd.DataFrame, model_name: str) -> dict:
    x = state_targets[["stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin"]].astype(float)
    y = state_targets[["targetLow", "targetHigh"]].astype(float)
    groups = state_targets["stateKey"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))

    if model_name == "hist_gbr":
        model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=220,
                learning_rate=0.05,
                random_state=42,
            )
        )
    elif model_name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(model_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x.iloc[train_idx], y.iloc[train_idx])

    pred = model.predict(x.iloc[test_idx])
    return {
        "model": model_name,
        "r2_low": float(r2_score(y.iloc[test_idx, 0], pred[:, 0])),
        "r2_high": float(r2_score(y.iloc[test_idx, 1], pred[:, 1])),
        "r2_uniform": float(r2_score(y.iloc[test_idx], pred, multioutput="uniform_average")),
        "mae_low": float(mean_absolute_error(y.iloc[test_idx, 0], pred[:, 0])),
        "mae_high": float(mean_absolute_error(y.iloc[test_idx, 1], pred[:, 1])),
        "train_states": int(len(set(groups.iloc[train_idx]))),
        "test_states": int(len(set(groups.iloc[test_idx]))),
        "unique_states": int(state_targets["stateKey"].nunique()),
    }


def main() -> None:
    config = ExperimentConfig(
        experiment_name="smoothed_policy_experiment",
        state_cbr_bin_size=5.0,
        state_neighbor_bin_size=5.0,
        min_pair_samples_per_state=10,
        include_run_pdr_in_label=False,
    )
    dataset = load_dataset()
    pair_score = build_pair_score(dataset, config)

    rows = []
    for bandwidth in [0.75, 1.0, 1.5, 2.0, 3.0]:
        for temperature in [0.5, 1.0, 2.0]:
            state_targets = build_smoothed_targets(pair_score, bandwidth=bandwidth, temperature=temperature)
            for model_name in ["hist_gbr", "extra_trees"]:
                row = evaluate_targets(state_targets, model_name)
                row["bandwidth"] = bandwidth
                row["temperature"] = temperature
                rows.append(row)
                print(
                    f"bandwidth={bandwidth}, temp={temperature}, model={model_name}, "
                    f"r2_uniform={row['r2_uniform']:.6f}"
                )

    result = pd.DataFrame(rows).sort_values("r2_uniform", ascending=False).reset_index(drop=True)
    output_path = OUTPUT_DIR / "smoothed_policy_experiment_results.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print()
    print(result.head(15).to_string(index=False))
    print()
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
