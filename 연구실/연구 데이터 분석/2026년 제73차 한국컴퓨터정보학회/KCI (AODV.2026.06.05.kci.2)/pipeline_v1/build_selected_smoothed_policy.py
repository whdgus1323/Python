from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor

from run_pipeline import ExperimentConfig, load_dataset
from experiment_soft_labels import build_pair_score
from experiment_smoothed_policy import build_smoothed_targets


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SELECTED_BANDWIDTH = 1.0
SELECTED_TEMPERATURE = 0.5


def evaluate_targets(state_targets: pd.DataFrame) -> pd.DataFrame:
    x = state_targets[["stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin"]].astype(float)
    y = state_targets[["targetLow", "targetHigh"]].astype(float)
    groups = state_targets["stateKey"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))

    rows = []
    models = {
        "dummy_mean": MultiOutputRegressor(DummyRegressor(strategy="mean")),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        rows.append(
            {
                "model": name,
                "r2_low": float(r2_score(y.iloc[test_idx, 0], pred[:, 0])),
                "r2_high": float(r2_score(y.iloc[test_idx, 1], pred[:, 1])),
                "r2_uniform": float(r2_score(y.iloc[test_idx], pred, multioutput="uniform_average")),
                "mae_low": float(mean_absolute_error(y.iloc[test_idx, 0], pred[:, 0])),
                "mae_high": float(mean_absolute_error(y.iloc[test_idx, 1], pred[:, 1])),
            }
        )
    return pd.DataFrame(rows).sort_values("r2_uniform", ascending=False).reset_index(drop=True)


def train_full_model(state_targets: pd.DataFrame) -> ExtraTreesRegressor:
    x = state_targets[["stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin"]].astype(float)
    y = state_targets[["targetLow", "targetHigh"]].astype(float)
    model = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y)
    return model


def add_rounded_policy_columns(state_targets: pd.DataFrame) -> pd.DataFrame:
    policy = state_targets.copy()
    policy["policyLow"] = policy["targetLow"].round().astype(int)
    policy["policyHigh"] = policy["targetHigh"].round().astype(int)
    return policy.sort_values(
        ["stateDirectBin", "stateHopBin", "stateCbrBin", "stateNeighborBin"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def build_full_lookup_grid(model: ExtraTreesRegressor) -> pd.DataFrame:
    rows = []
    cbr_bins = list(range(0, 101, 5))
    neighbor_bins = list(range(0, 101, 5))
    for state_cbr_bin in cbr_bins:
        for state_neighbor_bin in neighbor_bins:
            for state_hop_bin in [0, 1]:
                rows.append(
                    {
                        "stateCbrBin": state_cbr_bin,
                        "stateNeighborBin": state_neighbor_bin,
                        "stateHopBin": state_hop_bin,
                        "stateDirectBin": 1,
                    }
                )
            for state_hop_bin in [2, 3, 4, 5, 6]:
                rows.append(
                    {
                        "stateCbrBin": state_cbr_bin,
                        "stateNeighborBin": state_neighbor_bin,
                        "stateHopBin": state_hop_bin,
                        "stateDirectBin": 0,
                    }
                )

    grid = pd.DataFrame(rows)
    pred = model.predict(grid[["stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin"]].astype(float))
    grid["targetLowPred"] = pred[:, 0]
    grid["targetHighPred"] = pred[:, 1]
    grid["policyLow"] = grid["targetLowPred"].round().astype(int)
    grid["policyHigh"] = grid["targetHighPred"].round().astype(int)
    grid["stateKey"] = (
        grid["stateCbrBin"].astype(int).astype(str)
        + "|"
        + grid["stateNeighborBin"].astype(int).astype(str)
        + "|"
        + grid["stateHopBin"].astype(int).astype(str)
        + "|"
        + grid["stateDirectBin"].astype(int).astype(str)
    )
    return grid.sort_values(
        ["stateDirectBin", "stateHopBin", "stateCbrBin", "stateNeighborBin"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def main() -> None:
    config = ExperimentConfig(
        experiment_name="selected_smoothed_policy",
        state_cbr_bin_size=5.0,
        state_neighbor_bin_size=5.0,
        min_pair_samples_per_state=10,
        include_run_pdr_in_label=False,
    )
    dataset = load_dataset()
    pair_score = build_pair_score(dataset, config)
    state_targets = build_smoothed_targets(
        pair_score,
        bandwidth=SELECTED_BANDWIDTH,
        temperature=SELECTED_TEMPERATURE,
    )
    policy = add_rounded_policy_columns(state_targets)
    eval_table = evaluate_targets(state_targets)
    full_model = train_full_model(state_targets)
    full_lookup = build_full_lookup_grid(full_model)

    policy_path = OUTPUT_DIR / "selected_smoothed_policy_table.csv"
    full_lookup_path = OUTPUT_DIR / "selected_smoothed_policy_full_lookup.csv"
    summary_path = OUTPUT_DIR / "selected_smoothed_policy_eval.csv"
    meta_path = OUTPUT_DIR / "selected_smoothed_policy_meta.json"

    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    full_lookup.to_csv(full_lookup_path, index=False, encoding="utf-8-sig")
    eval_table.to_csv(summary_path, index=False, encoding="utf-8-sig")
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "bandwidth": SELECTED_BANDWIDTH,
                "temperature": SELECTED_TEMPERATURE,
                "unique_states": int(policy["stateKey"].nunique()),
                "uniqueRoundedPairs": int(policy[["policyLow", "policyHigh"]].drop_duplicates().shape[0]),
                "fullLookupRows": int(len(full_lookup)),
                "fullLookupUniqueRoundedPairs": int(full_lookup[["policyLow", "policyHigh"]].drop_duplicates().shape[0]),
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    print(eval_table.to_string(index=False))
    print()
    print(policy[["policyLow", "policyHigh"]].drop_duplicates().sort_values(["policyLow", "policyHigh"]).to_string(index=False))
    print()
    print(f"saved: {policy_path}")
    print(f"saved: {full_lookup_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {meta_path}")


if __name__ == "__main__":
    main()
