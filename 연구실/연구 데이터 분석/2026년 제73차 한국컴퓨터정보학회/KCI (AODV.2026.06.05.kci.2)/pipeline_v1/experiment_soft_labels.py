from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor

from run_pipeline import ExperimentConfig, build_labeled_dataset, load_dataset


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def build_pair_score(dataset: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    labeled = dataset.copy()
    labeled["stateCbrBin"] = np.clip(
        np.floor(labeled["localCbr"] / config.state_cbr_bin_size) * config.state_cbr_bin_size,
        0,
        100,
    )
    labeled["stateNeighborBin"] = np.clip(
        np.floor(labeled["neighborCount"] / config.state_neighbor_bin_size) * config.state_neighbor_bin_size,
        0,
        100,
    )
    labeled["stateHopBin"] = np.clip(labeled["hopCount"].round().astype(int), 0, 6)
    labeled["stateDirectBin"] = labeled["isDirectRoute"].round().astype(int)
    labeled["stateKey"] = (
        labeled["stateCbrBin"].astype(int).astype(str)
        + "|"
        + labeled["stateNeighborBin"].astype(int).astype(str)
        + "|"
        + labeled["stateHopBin"].astype(int).astype(str)
        + "|"
        + labeled["stateDirectBin"].astype(int).astype(str)
    )
    labeled["offlineScore"] = (
        2.0 * labeled["futureSucceeded"].fillna(0)
        - 3.0 * labeled["futureFailed"].fillna(0)
        - 0.02 * labeled["futureDelay"].fillna(0)
    )

    pair_score = (
        labeled.groupby(
            ["stateKey", "stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin", "lowInt", "highInt"],
            observed=False,
        )
        .agg(
            scoreMean=("offlineScore", "mean"),
            sampleCount=("offlineScore", "size"),
        )
        .reset_index()
    )
    return pair_score[pair_score["sampleCount"] >= config.min_pair_samples_per_state].copy()


def build_state_targets(pair_score: pd.DataFrame, label_mode: str, temperature: float) -> pd.DataFrame:
    if label_mode == "hard":
        state_targets = (
            pair_score.sort_values(["stateKey", "scoreMean", "sampleCount"], ascending=[True, False, False])
            .groupby("stateKey", observed=False)
            .head(1)
            .rename(columns={"lowInt": "targetLow", "highInt": "targetHigh"})
        )
        return state_targets[
            [
                "stateKey",
                "stateCbrBin",
                "stateNeighborBin",
                "stateHopBin",
                "stateDirectBin",
                "targetLow",
                "targetHigh",
                "sampleCount",
            ]
        ].rename(columns={"sampleCount": "stateWeight"})

    scored = pair_score.copy()
    scored["scoreShift"] = scored["scoreMean"] - scored.groupby("stateKey", observed=False)["scoreMean"].transform("max")
    scored["softWeight"] = np.exp(scored["scoreShift"] / temperature) * scored["sampleCount"].clip(lower=1)
    scored["weightedLow"] = scored["softWeight"] * scored["lowInt"]
    scored["weightedHigh"] = scored["softWeight"] * scored["highInt"]

    state_targets = (
        scored.groupby(["stateKey", "stateCbrBin", "stateNeighborBin", "stateHopBin", "stateDirectBin"], observed=False)
        .agg(
            targetLow=("weightedLow", lambda s: float(s.sum())),
            targetHigh=("weightedHigh", lambda s: float(s.sum())),
            softWeightSum=("softWeight", lambda s: float(s.sum())),
            stateWeight=("sampleCount", "sum"),
        )
        .reset_index()
    )
    state_targets["targetLow"] = state_targets["targetLow"] / state_targets["softWeightSum"]
    state_targets["targetHigh"] = state_targets["targetHigh"] / state_targets["softWeightSum"]
    return state_targets.drop(columns=["softWeightSum"])


def evaluate_state_level(state_targets: pd.DataFrame, model_name: str) -> dict:
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
    elif model_name == "random_forest":
        model = RandomForestRegressor(
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
        experiment_name="soft_label_experiment",
        state_cbr_bin_size=5.0,
        state_neighbor_bin_size=5.0,
        min_pair_samples_per_state=10,
        include_run_pdr_in_label=False,
    )
    dataset = load_dataset()
    labeled_dataset = build_labeled_dataset(dataset, config)
    pair_score = build_pair_score(dataset, config)

    rows = []
    for label_mode, temperature in [
        ("hard", 1.0),
        ("soft", 0.25),
        ("soft", 0.5),
        ("soft", 1.0),
        ("soft", 2.0),
    ]:
        state_targets = build_state_targets(pair_score, label_mode=label_mode, temperature=temperature)
        for model_name in ["hist_gbr", "extra_trees", "random_forest"]:
            row = evaluate_state_level(state_targets, model_name)
            row["label_mode"] = label_mode
            row["temperature"] = temperature
            row["row_reference_count"] = len(labeled_dataset)
            rows.append(row)
            print(
                f"label={label_mode}, temp={temperature}, model={model_name}, "
                f"r2_uniform={row['r2_uniform']:.6f}"
            )

    result = pd.DataFrame(rows).sort_values("r2_uniform", ascending=False).reset_index(drop=True)
    output_path = OUTPUT_DIR / "soft_label_experiment_results.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print()
    print(result.head(12).to_string(index=False))
    print()
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
