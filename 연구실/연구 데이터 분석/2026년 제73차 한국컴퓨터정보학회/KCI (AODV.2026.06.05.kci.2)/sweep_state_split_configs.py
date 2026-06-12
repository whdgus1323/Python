from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor


random_base_root = Path(r"C:\Users\Choe JongHyeon\Desktop\map_v2\random_base1")
run_ids = [4, 10, 14, 15, 16]
seed_name = "seed_1"

random_state = 42
chunk_size = 250_000
max_rows_per_run = 60_000

neighbor_norm = 20.0
hop_norm = 10.0
future_window_sec = 2
hop_bin_max = 6

pdr_pattern = re.compile(r"PDR\s*\(%\):\s*([0-9.]+)")


def parse_run_pdr(result_path: Path) -> float:
    if not result_path.exists():
        return np.nan
    text = result_path.read_text(encoding="utf-8", errors="ignore")
    match = pdr_pattern.search(text)
    return float(match.group(1)) if match else np.nan


def build_future_table_for_run(run_id: int) -> pd.DataFrame:
    base = random_base_root / str(run_id) / seed_name
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
        for offset in range(future_window_sec + 1):
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
    base = random_base_root / str(run_id) / seed_name
    future_table = build_future_table_for_run(run_id)
    decision_path = base / "aodv_cbr_rrep_decisions.csv"

    parts = []
    total = 0
    for chunk_index, chunk in enumerate(
        pd.read_csv(decision_path, usecols=["time", "node", "hopCount"], chunksize=chunk_size),
        start=1,
    ):
        chunk = chunk.dropna(subset=["time", "node", "hopCount"]).copy()
        if chunk.empty:
            continue

        remaining = max_rows_per_run - total
        if remaining <= 0:
            break

        take = min(remaining, max(1, int(len(chunk) * 0.03)))
        if take < len(chunk):
            chunk = chunk.sample(n=take, random_state=random_state + run_id + chunk_index)

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


def load_base_dataset() -> pd.DataFrame:
    frames = [load_decision_sample_for_run(run_id) for run_id in run_ids]
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


def build_labeled_dataset(
    dataset: pd.DataFrame,
    state_cbr_bin_size: float,
    state_neighbor_bin_size: float,
    min_pair_samples_per_state: int,
    include_run_pdr: bool,
) -> pd.DataFrame:
    labeled = dataset.copy()
    labeled["stateCbrBin"] = np.clip(
        np.floor(labeled["localCbr"] / state_cbr_bin_size) * state_cbr_bin_size,
        0,
        100,
    )
    labeled["stateNeighborBin"] = np.clip(
        np.floor(labeled["neighborCount"] / state_neighbor_bin_size) * state_neighbor_bin_size,
        0,
        100,
    )
    labeled["stateHopBin"] = np.clip(labeled["hopCount"].round().astype(int), 0, hop_bin_max)
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
    if include_run_pdr:
        score = score + labeled["runPdr"].fillna(0)

    labeled["offlineScore"] = score

    pair_score = (
        labeled.groupby(["stateKey", "lowInt", "highInt"], observed=False)
        .agg(
            scoreMean=("offlineScore", "mean"),
            sampleCount=("offlineScore", "size"),
        )
        .reset_index()
    )
    pair_score = pair_score[pair_score["sampleCount"] >= min_pair_samples_per_state].copy()
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


def evaluate_config(labeled_dataset: pd.DataFrame, config_name: str) -> dict:
    x = pd.DataFrame(
        {
            "localCbrNorm": np.clip(labeled_dataset["localCbr"].astype(float) / 100.0, 0, 1),
            "neighborNorm": np.clip(labeled_dataset["neighborCount"].astype(float) / neighbor_norm, 0, 1),
            "hopNorm": np.clip(labeled_dataset["hopCount"].astype(float) / hop_norm, 0, 1),
            "isDirectRoute": labeled_dataset["isDirectRoute"].astype(float),
        }
    )
    y = labeled_dataset[["bestLow", "bestHigh"]].astype(float)
    groups = labeled_dataset["stateKey"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))

    train_groups = set(groups.iloc[train_idx].unique())
    test_groups = set(groups.iloc[test_idx].unique())
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"{config_name}: stateKey leakage detected: {len(overlap)}")

    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=260,
            learning_rate=0.05,
            random_state=random_state,
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x.iloc[train_idx], y.iloc[train_idx])

    pred = model.predict(x.iloc[test_idx])
    y_test = y.iloc[test_idx]
    r2_each = r2_score(y_test, pred, multioutput="raw_values")
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

    return {
        "config_name": config_name,
        "labeled_rows": len(labeled_dataset),
        "unique_states": labeled_dataset["stateKey"].nunique(),
        "unique_best_pairs": labeled_dataset[["bestLow", "bestHigh"]].drop_duplicates().shape[0],
        "train_states": len(train_groups),
        "test_states": len(test_groups),
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


def main() -> None:
    base_dataset = load_base_dataset()
    configs = []
    for include_run_pdr in [True, False]:
        for bin_size in [5.0, 10.0, 20.0]:
            for min_pair_samples in [3, 5, 10]:
                configs.append(
                    {
                        "include_run_pdr": include_run_pdr,
                        "state_cbr_bin_size": bin_size,
                        "state_neighbor_bin_size": bin_size,
                        "min_pair_samples_per_state": min_pair_samples,
                    }
                )

    rows = []
    for cfg in configs:
        config_name = (
            f"runpdr_{int(cfg['include_run_pdr'])}"
            f"_bin_{cfg['state_cbr_bin_size']:.0f}"
            f"_min_{cfg['min_pair_samples_per_state']}"
        )
        labeled_dataset = build_labeled_dataset(
            dataset=base_dataset,
            state_cbr_bin_size=cfg["state_cbr_bin_size"],
            state_neighbor_bin_size=cfg["state_neighbor_bin_size"],
            min_pair_samples_per_state=cfg["min_pair_samples_per_state"],
            include_run_pdr=cfg["include_run_pdr"],
        )
        row = evaluate_config(labeled_dataset, config_name)
        row.update(cfg)
        rows.append(row)
        print(
            f"{config_name}: r2_uniform={row['r2_uniform']:.6f}, "
            f"states={row['unique_states']}, rows={row['labeled_rows']}"
        )

    result = pd.DataFrame(rows).sort_values("r2_uniform", ascending=False).reset_index(drop=True)
    output_path = Path(__file__).with_name("sweep_state_split_results.csv")
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print()
    print(result.head(10).to_string(index=False))
    print()
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
