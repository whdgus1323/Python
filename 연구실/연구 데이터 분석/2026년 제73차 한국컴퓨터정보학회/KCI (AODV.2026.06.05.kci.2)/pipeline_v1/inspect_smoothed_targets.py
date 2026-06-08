from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_pipeline import ExperimentConfig, load_dataset
from experiment_soft_labels import build_pair_score
from experiment_smoothed_policy import build_smoothed_targets


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    config = ExperimentConfig(
        experiment_name="inspect_smoothed_targets",
        state_cbr_bin_size=5.0,
        state_neighbor_bin_size=5.0,
        min_pair_samples_per_state=10,
        include_run_pdr_in_label=False,
    )
    dataset = load_dataset()
    pair_score = build_pair_score(dataset, config)

    rows = []
    for bandwidth, temperature in [
        (0.75, 0.5),
        (0.75, 1.0),
        (0.75, 2.0),
        (1.0, 0.5),
        (1.0, 1.0),
        (1.0, 2.0),
        (1.5, 0.5),
        (2.0, 0.5),
        (3.0, 0.5),
    ]:
        targets = build_smoothed_targets(pair_score, bandwidth=bandwidth, temperature=temperature)
        rounded = targets.copy()
        rounded["roundLow"] = rounded["targetLow"].round().astype(int)
        rounded["roundHigh"] = rounded["targetHigh"].round().astype(int)
        rows.append(
            {
                "bandwidth": bandwidth,
                "temperature": temperature,
                "targetLow_min": float(targets["targetLow"].min()),
                "targetLow_max": float(targets["targetLow"].max()),
                "targetLow_std": float(targets["targetLow"].std()),
                "targetHigh_min": float(targets["targetHigh"].min()),
                "targetHigh_max": float(targets["targetHigh"].max()),
                "targetHigh_std": float(targets["targetHigh"].std()),
                "uniqueRoundedPairs": int(rounded[["roundLow", "roundHigh"]].drop_duplicates().shape[0]),
            }
        )
        rounded.to_csv(
            OUTPUT_DIR / f"smoothed_targets_bw{str(bandwidth).replace('.', '_')}_temp{str(temperature).replace('.', '_')}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "smoothed_target_variation_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print()
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
