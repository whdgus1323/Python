from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_BASE_ROOT = Path(r"C:\Users\Choe JongHyeon\Desktop\map_v2\random_base1")
RUN_IDS = [4, 10, 14, 15, 16]
SEED_NAME = "seed_1"

RANDOM_STATE = 42
CHUNK_SIZE = 250_000
MAX_ROWS_PER_RUN = 60_000

NEIGHBOR_NORM = 20.0
HOP_NORM = 10.0
FUTURE_WINDOW_SEC = 2
HOP_BIN_MAX = 6
PDR_PATTERN = re.compile(r"PDR\s*\(%\):\s*([0-9.]+)")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "pipeline_v1_default"
    state_cbr_bin_size: float = 5.0
    state_neighbor_bin_size: float = 5.0
    min_pair_samples_per_state: int = 10
    include_run_pdr_in_label: bool = False
    test_size: float = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "model"], default="baseline")
    parser.add_argument("--state-cbr-bin", type=float, default=5.0)
    parser.add_argument("--state-neighbor-bin", type=float, default=5.0)
    parser.add_argument("--min-pair-samples", type=int, default=10)
    parser.add_argument("--include-run-pdr", action="store_true")
    return parser.parse_args()


def parse_run_pdr(result_path: Path) -> float:
    if not result_path.exists():
        return np.nan
    text = result_path.read_text(encoding="utf-8", errors="ignore")
    match = PDR_PATTERN.search(text)
    return float(match.group(1)) if match else np.nan


def build_future_table_for_run(run_id: int) -> pd.DataFrame:
    base = RANDOM_BASE_ROOT / str(run_id) / SEED_NAME
    path = base / "aodv_transmission_failure_diagnosis_1s.csv"
    dia = pd.read_csv(
        path,
        usecols=[
            "time",
            "node",
            "localCbr",
            "neighborCount",
            "appliedLowThreshold",
            "appliedHighThreshold",
            "routeDiscoveryStarted",
            "routeDiscoverySucceeded",
            "routeDiscoveryFailed",
            "routeDiscoveryDelayAvgMs",
        ],
    )

    dia["sec"] = pd.to_numeric(dia["time"], errors="coerce").astype("Int64")
    for column in [
        "localCbr",
        "neighborCount",
        "appliedLowThreshold",
        "appliedHighThreshold",
        "routeDiscoveryStarted",
        "routeDiscoverySucceeded",
        "routeDiscoveryFailed",
        "routeDiscoveryDelayAvgMs",
    ]:
        dia[column] = pd.to_numeric(dia[column], errors="coerce").fillna(0.0)

    dia = dia.dropna(subset=["sec", "node"]).drop_duplicates(["sec", "node"]).copy()
    lookup = dia.set_index(["sec", "node"])[
        [
            "routeDiscoveryStarted",
            "routeDiscoverySucceeded",
            "routeDiscoveryFailed",
            "routeDiscoveryDelayAvgMs",
        ]
    ]

    future_rows = []
    for sec, node in dia[["sec", "node"]].itertuples(index=False):
        started_sum = 0.0
        succeeded_sum = 0.0
        failed_sum = 0.0
        delay_values = []
        for offset in range(FUTURE_WINDOW_SEC + 1):
            try:
                values = lookup.loc[(int(sec) + offset, node)]
            except KeyError:
                continue
            started_sum += float(values["routeDiscoveryStarted"])
            succeeded_sum += float(values["routeDiscoverySucceeded"])
            failed_sum += float(values["routeDiscoveryFailed"])
            delay_value = float(values["routeDiscoveryDelayAvgMs"])
            if delay_value > 0:
                delay_values.append(delay_value)

        future_rows.append(
            {
                "sec": int(sec),
                "node": node,
                "futureStarted": started_sum,
                "futureSucceeded": succeeded_sum,
                "futureFailed": failed_sum,
                "futureDelay": float(np.mean(delay_values)) if delay_values else 0.0,
            }
        )

    future_df = pd.DataFrame(future_rows)
    base_state = dia[
        [
            "sec",
            "node",
            "localCbr",
            "neighborCount",
            "appliedLowThreshold",
            "appliedHighThreshold",
        ]
    ].copy()
    return base_state.merge(future_df, on=["sec", "node"], how="left")


def load_decision_sample_for_run(run_id: int) -> pd.DataFrame:
    base = RANDOM_BASE_ROOT / str(run_id) / SEED_NAME
    future_table = build_future_table_for_run(run_id)
    decision_path = base / "aodv_cbr_rrep_decisions.csv"

    parts = []
    total = 0
    for chunk_index, chunk in enumerate(
        pd.read_csv(decision_path, usecols=["time", "node", "hopCount"], chunksize=CHUNK_SIZE),
        start=1,
    ):
        chunk = chunk.dropna(subset=["time", "node", "hopCount"]).copy()
        if chunk.empty:
            continue

        remaining = MAX_ROWS_PER_RUN - total
        if remaining <= 0:
            break

        take = min(remaining, max(1, int(len(chunk) * 0.03)))
        if take < len(chunk):
            chunk = chunk.sample(n=take, random_state=RANDOM_STATE + run_id + chunk_index)

        chunk["sec"] = np.floor(pd.to_numeric(chunk["time"], errors="coerce")).astype("Int64")
        chunk["isDirectRoute"] = (pd.to_numeric(chunk["hopCount"], errors="coerce") <= 1).astype(float)
        parts.append(chunk)
        total += len(chunk)

    decision_sample = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if decision_sample.empty:
        return decision_sample

    decision_sample = decision_sample.merge(future_table, on=["sec", "node"], how="left")
    decision_sample["run"] = run_id
    decision_sample["runPdr"] = parse_run_pdr(base / "result.txt")
    return decision_sample


def load_dataset() -> pd.DataFrame:
    frames = [load_decision_sample_for_run(run_id) for run_id in RUN_IDS]
    dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if dataset.empty:
        raise ValueError("dataset is empty. Check random_base_root and completed runs.")

    for column in [
        "localCbr",
        "neighborCount",
        "hopCount",
        "isDirectRoute",
        "appliedLowThreshold",
        "appliedHighThreshold",
        "futureStarted",
        "futureSucceeded",
        "futureFailed",
        "futureDelay",
        "runPdr",
    ]:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    dataset = dataset.dropna(
        subset=[
            "localCbr",
            "neighborCount",
            "hopCount",
            "isDirectRoute",
            "appliedLowThreshold",
            "appliedHighThreshold",
        ]
    ).copy()
    dataset["lowInt"] = dataset["appliedLowThreshold"].round().astype(int)
    dataset["highInt"] = dataset["appliedHighThreshold"].round().astype(int)
    return dataset


def build_labeled_dataset(dataset: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
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
    labeled["stateHopBin"] = np.clip(labeled["hopCount"].round().astype(int), 0, HOP_BIN_MAX)
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

    score = (
        2.0 * labeled["futureSucceeded"].fillna(0)
        - 3.0 * labeled["futureFailed"].fillna(0)
        - 0.02 * labeled["futureDelay"].fillna(0)
    )
    if config.include_run_pdr_in_label:
        score = score + labeled["runPdr"].fillna(0)
    labeled["offlineScore"] = score

    pair_score = (
        labeled.groupby(["stateKey", "lowInt", "highInt"], observed=False)
        .agg(scoreMean=("offlineScore", "mean"), sampleCount=("offlineScore", "size"))
        .reset_index()
    )
    pair_score = pair_score[pair_score["sampleCount"] >= config.min_pair_samples_per_state].copy()
    pair_score = pair_score.sort_values(
        ["stateKey", "scoreMean", "sampleCount"],
        ascending=[True, False, False],
    )

    best_pair_by_state = (
        pair_score.groupby("stateKey", observed=False)
        .head(1)
        .rename(
            columns={
                "lowInt": "bestLow",
                "highInt": "bestHigh",
                "scoreMean": "bestScore",
                "sampleCount": "bestPairSampleCount",
            }
        )
    )
    return labeled.merge(
        best_pair_by_state[["stateKey", "bestLow", "bestHigh", "bestScore", "bestPairSampleCount"]],
        on="stateKey",
        how="inner",
    ).copy()


def make_xy(labeled_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    x = pd.DataFrame(
        {
            "localCbrNorm": np.clip(labeled_dataset["localCbr"].astype(float) / 100.0, 0, 1),
            "neighborNorm": np.clip(labeled_dataset["neighborCount"].astype(float) / NEIGHBOR_NORM, 0, 1),
            "hopNorm": np.clip(labeled_dataset["hopCount"].astype(float) / HOP_NORM, 0, 1),
            "isDirectRoute": labeled_dataset["isDirectRoute"].astype(float),
        }
    )
    y = labeled_dataset[["bestLow", "bestHigh"]].astype(float)
    groups = labeled_dataset["stateKey"].astype(str)
    return x, y, groups


def split_groups(x: pd.DataFrame, y: pd.DataFrame, groups: pd.Series, test_size: float) -> dict:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))

    train_groups = set(groups.iloc[train_idx].unique())
    test_groups = set(groups.iloc[test_idx].unique())
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"stateKey leakage detected: {len(overlap)} overlapping states")

    return {
        "x_train": x.iloc[train_idx].reset_index(drop=True),
        "x_test": x.iloc[test_idx].reset_index(drop=True),
        "y_train": y.iloc[train_idx].reset_index(drop=True),
        "y_test": y.iloc[test_idx].reset_index(drop=True),
        "train_states": len(train_groups),
        "test_states": len(test_groups),
        "overlap_states": len(overlap),
    }


def score_prediction(y_test: pd.DataFrame, pred: np.ndarray) -> dict:
    preview = pd.DataFrame(
        {
            "actualLow": y_test.iloc[:, 0].to_numpy(),
            "actualHigh": y_test.iloc[:, 1].to_numpy(),
            "predLow": pred[:, 0],
            "predHigh": pred[:, 1],
        }
    )
    preview["pairExactMatch"] = (
        preview["predLow"].round().astype(int).eq(preview["actualLow"].round().astype(int))
        & preview["predHigh"].round().astype(int).eq(preview["actualHigh"].round().astype(int))
    )
    preview["lowError"] = np.abs(preview["predLow"] - preview["actualLow"])
    preview["highError"] = np.abs(preview["predHigh"] - preview["actualHigh"])

    r2_each = r2_score(y_test, pred, multioutput="raw_values")
    return {
        "r2_low": float(r2_each[0]),
        "r2_high": float(r2_each[1]),
        "r2_uniform": float(r2_score(y_test, pred, multioutput="uniform_average")),
        "mae_low": float(mean_absolute_error(y_test.iloc[:, 0], pred[:, 0])),
        "mae_high": float(mean_absolute_error(y_test.iloc[:, 1], pred[:, 1])),
        "rmse_low": float(np.sqrt(mean_squared_error(y_test.iloc[:, 0], pred[:, 0]))),
        "rmse_high": float(np.sqrt(mean_squared_error(y_test.iloc[:, 1], pred[:, 1]))),
        "pair_exact_match_rate": float(preview["pairExactMatch"].mean()),
        "mean_low_error": float(preview["lowError"].mean()),
        "mean_high_error": float(preview["highError"].mean()),
    }


def run_baseline(split: dict) -> pd.DataFrame:
    rows = []
    for strategy in ["mean", "median"]:
        model = MultiOutputRegressor(DummyRegressor(strategy=strategy))
        model.fit(split["x_train"], split["y_train"])
        pred = model.predict(split["x_test"])
        row = {"model": f"dummy_{strategy}"}
        row.update(score_prediction(split["y_test"], pred))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("r2_uniform", ascending=False).reset_index(drop=True)


def run_model(split: dict) -> pd.DataFrame:
    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(split["x_train"], split["y_train"])
    pred = model.predict(split["x_test"])
    row = {"model": "hist_gradient_boosting"}
    row.update(score_prediction(split["y_test"], pred))
    return pd.DataFrame([row])


def save_outputs(config: ExperimentConfig, labeled_dataset: pd.DataFrame, split: dict, result: pd.DataFrame, mode: str) -> None:
    stamp = f"{mode}_{config.experiment_name}"
    summary = result.copy()
    summary["labeled_rows"] = len(labeled_dataset)
    summary["unique_states"] = labeled_dataset["stateKey"].nunique()
    summary["unique_best_pairs"] = labeled_dataset[["bestLow", "bestHigh"]].drop_duplicates().shape[0]
    summary["train_states"] = split["train_states"]
    summary["test_states"] = split["test_states"]
    summary["overlap_states"] = split["overlap_states"]
    summary["config_json"] = json.dumps(asdict(config), ensure_ascii=False)
    summary.to_csv(OUTPUT_DIR / f"summary_{stamp}.csv", index=False, encoding="utf-8-sig")

    with (OUTPUT_DIR / f"config_{stamp}.json").open("w", encoding="utf-8") as fp:
        json.dump(asdict(config), fp, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        experiment_name=(
            f"mode_{args.mode}"
            f"_cbr{int(args.state_cbr_bin)}"
            f"_nbr{int(args.state_neighbor_bin)}"
            f"_min{args.min_pair_samples}"
            f"_runpdr{int(args.include_run_pdr)}"
        ),
        state_cbr_bin_size=args.state_cbr_bin,
        state_neighbor_bin_size=args.state_neighbor_bin,
        min_pair_samples_per_state=args.min_pair_samples,
        include_run_pdr_in_label=args.include_run_pdr,
    )

    dataset = load_dataset()
    labeled_dataset = build_labeled_dataset(dataset, config)
    x, y, groups = make_xy(labeled_dataset)
    split = split_groups(x, y, groups, config.test_size)

    if args.mode == "baseline":
        result = run_baseline(split)
    else:
        result = run_model(split)

    save_outputs(config, labeled_dataset, split, result, args.mode)
    print(result.to_string(index=False))
    print()
    print(
        json.dumps(
            {
                "labeled_rows": len(labeled_dataset),
                "unique_states": int(labeled_dataset["stateKey"].nunique()),
                "train_states": split["train_states"],
                "test_states": split["test_states"],
                "overlap_states": split["overlap_states"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
