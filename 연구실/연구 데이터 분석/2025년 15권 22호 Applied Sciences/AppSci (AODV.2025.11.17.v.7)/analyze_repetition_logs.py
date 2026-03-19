#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


KV_PATTERN = re.compile(r"([A-Za-z_]+)=([^,]+)")
RERR_HEADER_PATTERN = re.compile(r"\[RERR 생성\] 시간: ([0-9.]+), 노드: ([^,]+), 목적지 수: (\d+)")
PRECURSOR_HEADER_PATTERN = re.compile(r"\[RERR 생성\] 시간: ([0-9.]+), 노드: ([^,]+), Precursor 수: (\d+)")
RESULT_SENT_PATTERN = re.compile(r"sent:\s*(\d+)")
RESULT_RECEIVED_PATTERN = re.compile(r"received:\s*(\d+)")
RESULT_PDR_PATTERN = re.compile(r"PDR \(%\):\s*([0-9.]+)")


def parse_kv_line(line: str) -> dict[str, str]:
    return {key: value.strip() for key, value in KV_PATTERN.findall(line)}


def bucket_time(value: float, bucket_size: float) -> float:
    return math.floor(value / bucket_size) * bucket_size


def parse_control_log(path: Path, bucket_size: float) -> tuple[list[dict[str, object]], Counter]:
    rows: list[dict[str, object]] = []
    counts: Counter = Counter()
    if not path.exists():
        return rows, counts
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            entry = parse_kv_line(line)
            if "time" not in entry or "event" not in entry:
                continue
            time_value = float(entry["time"])
            bucket = bucket_time(time_value, bucket_size)
            event = entry["event"]
            counts[(bucket, event)] += 1
            rows.append({"time": time_value, "bucket": bucket, "event": event, **entry})
    return rows, counts


def parse_contention_log(path: Path, bucket_size: float) -> tuple[list[dict[str, object]], Counter, defaultdict]:
    rows: list[dict[str, object]] = []
    counts: Counter = Counter()
    metrics: defaultdict = defaultdict(list)
    if not path.exists():
        return rows, counts, metrics
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            entry = parse_kv_line(line)
            if "time" not in entry:
                continue
            time_value = float(entry["time"])
            bucket = bucket_time(time_value, bucket_size)
            if "event" in entry:
                event = entry["event"]
                counts[(bucket, event)] += 1
                if event == "CHANNEL_GRANTED" and "contentionTime" in entry:
                    metrics[(bucket, "contentionTime")].append(float(entry["contentionTime"].rstrip("s")))
                if event == "SCHEDULE_TX" and "waitInterval" in entry:
                    metrics[(bucket, "waitInterval")].append(float(entry["waitInterval"].rstrip("s")))
                rows.append({"time": time_value, "bucket": bucket, "event": event, **entry})
            else:
                medium_busy = int(entry.get("medium_busy", "0"))
                medium_free = int(entry.get("medium_free", "0"))
                backoff_start = int(entry.get("backoff_start", "0"))
                schedule_tx = int(entry.get("schedule_tx", "0"))
                channel_granted = int(entry.get("channel_granted", "0"))
                counts[(bucket, "MEDIUM_BUSY")] += medium_busy
                counts[(bucket, "MEDIUM_FREE")] += medium_free
                counts[(bucket, "BACKOFF_START")] += backoff_start
                counts[(bucket, "SCHEDULE_TX")] += schedule_tx
                counts[(bucket, "CHANNEL_GRANTED")] += channel_granted
                if "avg_contention_time_s" in entry and channel_granted > 0:
                    metrics[(bucket, "contentionTime")].extend([float(entry["avg_contention_time_s"])] * channel_granted)
                if "avg_wait_interval_s" in entry and schedule_tx > 0:
                    metrics[(bucket, "waitInterval")].extend([float(entry["avg_wait_interval_s"])] * schedule_tx)
                rows.append({"time": time_value, "bucket": bucket, **entry})
    return rows, counts, metrics


def parse_rerr_debug(path: Path, bucket_size: float) -> tuple[list[dict[str, object]], Counter]:
    events: list[dict[str, object]] = []
    counts: Counter = Counter()
    if not path.exists():
        return events, counts
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = RERR_HEADER_PATTERN.match(line)
            if not match:
                continue
            time_value = float(match.group(1))
            node = match.group(2)
            destinations = int(match.group(3))
            bucket = bucket_time(time_value, bucket_size)
            counts[(bucket, "RERR_ORIGINATED")] += 1
            counts[(bucket, "RERR_DESTINATIONS")] += destinations
            events.append({
                "time": time_value,
                "bucket": bucket,
                "node": node,
                "destinations": destinations,
            })
    return events, counts


def parse_rerr_precursors(path: Path, bucket_size: float) -> tuple[list[dict[str, object]], Counter]:
    events: list[dict[str, object]] = []
    counts: Counter = Counter()
    if not path.exists():
        return events, counts
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            match = PRECURSOR_HEADER_PATTERN.match(line)
            if not match:
                continue
            time_value = float(match.group(1))
            node = match.group(2)
            precursors = int(match.group(3))
            bucket = bucket_time(time_value, bucket_size)
            counts[(bucket, "RERR_PRECURSORS_TOTAL")] += precursors
            events.append({
                "time": time_value,
                "bucket": bucket,
                "node": node,
                "precursors": precursors,
            })
    return events, counts


def parse_result(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    sent = received = None
    pdr = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if sent is None:
                match = RESULT_SENT_PATTERN.search(line)
                if match:
                    sent = int(match.group(1))
            if received is None:
                match = RESULT_RECEIVED_PATTERN.search(line)
                if match:
                    received = int(match.group(1))
            if pdr is None:
                match = RESULT_PDR_PATTERN.search(line)
                if match:
                    pdr = float(match.group(1))
    result: dict[str, float] = {}
    if sent is not None:
        result["sent"] = sent
    if received is not None:
        result["received"] = received
    if pdr is not None:
        result["pdr_percent"] = pdr
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_timeseries(
    bucket_size: float,
    control_counts: Counter,
    contention_counts: Counter,
    contention_metrics: defaultdict,
    rerr_counts: Counter,
    precursor_counts: Counter,
) -> list[dict[str, object]]:
    buckets = sorted({
        bucket
        for bucket, _ in list(control_counts.keys())
        + list(contention_counts.keys())
        + list(contention_metrics.keys())
        + list(rerr_counts.keys())
        + list(precursor_counts.keys())
    })
    rows: list[dict[str, object]] = []
    for bucket in buckets:
        def count(source: Counter, name: str) -> int:
            return source[(bucket, name)]

        def avg(metric_name: str) -> float:
            values = contention_metrics.get((bucket, metric_name), [])
            return sum(values) / len(values) if values else 0.0

        rows.append({
            "time_bucket_s": bucket,
            "bucket_size_s": bucket_size,
            "rreq_send": count(control_counts, "RREQ_SEND"),
            "rreq_recv": count(control_counts, "RREQ_RECV"),
            "rreq_duplicate_drop": count(control_counts, "RREQ_DUPLICATE_DROP"),
            "rrep_send": count(control_counts, "RREP_SEND"),
            "rrep_recv": count(control_counts, "RREP_RECV"),
            "rrep_timeout": count(control_counts, "RREP_TIMEOUT"),
            "rerr_recv": count(control_counts, "RERR_RECV"),
            "rerr_originated": count(rerr_counts, "RERR_ORIGINATED"),
            "rerr_destinations_total": count(rerr_counts, "RERR_DESTINATIONS"),
            "rerr_precursors_total": count(precursor_counts, "RERR_PRECURSORS_TOTAL"),
            "medium_busy": count(contention_counts, "MEDIUM_BUSY"),
            "medium_free": count(contention_counts, "MEDIUM_FREE"),
            "backoff_start": count(contention_counts, "BACKOFF_START"),
            "schedule_tx": count(contention_counts, "SCHEDULE_TX"),
            "channel_granted": count(contention_counts, "CHANNEL_GRANTED"),
            "avg_contention_time_s": avg("contentionTime"),
            "avg_wait_interval_s": avg("waitInterval"),
        })
    return rows


def try_plot(timeseries_path: Path, output_dir: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    rows = []
    with timeseries_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        return []

    time_values = [float(row["time_bucket_s"]) for row in rows]
    graph_paths: list[Path] = []

    def save_plot(file_name: str, series: list[tuple[str, list[float]]], title: str, ylabel: str) -> None:
        plt.figure(figsize=(10, 4.5))
        for label, values in series:
            plt.plot(time_values, values, marker="o", linewidth=1.8, label=label)
        plt.title(title)
        plt.xlabel("time (s)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        graph_path = output_dir / file_name
        plt.savefig(graph_path, dpi=160)
        plt.close()
        graph_paths.append(graph_path)

    save_plot(
        "control_plane_events.png",
        [
            ("RREQ send", [float(row["rreq_send"]) for row in rows]),
            ("RREQ duplicate drop", [float(row["rreq_duplicate_drop"]) for row in rows]),
            ("RREP recv", [float(row["rrep_recv"]) for row in rows]),
            ("RREP timeout", [float(row["rrep_timeout"]) for row in rows]),
            ("RERR originated", [float(row["rerr_originated"]) for row in rows]),
        ],
        "AODV Control Plane Events",
        "count per bucket",
    )

    save_plot(
        "contention_pressure.png",
        [
            ("Medium busy", [float(row["medium_busy"]) for row in rows]),
            ("Backoff start", [float(row["backoff_start"]) for row in rows]),
            ("Channel granted", [float(row["channel_granted"]) for row in rows]),
        ],
        "MAC Contention Pressure",
        "count per bucket",
    )

    save_plot(
        "contention_delay.png",
        [
            ("Avg contention time", [float(row["avg_contention_time_s"]) for row in rows]),
            ("Avg wait interval", [float(row["avg_wait_interval_s"]) for row in rows]),
        ],
        "MAC Contention Delay",
        "seconds",
    )
    return graph_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize AODV/MAC repetition logs into CSV and optional graphs.")
    parser.add_argument("log_dir", type=Path, help="Directory pointed to by **.pwd")
    parser.add_argument("--bucket-size", type=float, default=1.0, help="Time bucket size in seconds")
    args = parser.parse_args()

    log_dir = args.log_dir
    out_dir = log_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    control_rows, control_counts = parse_control_log(log_dir / "aodv_control_log.txt", args.bucket_size)
    contention_rows, contention_counts, contention_metrics = parse_contention_log(log_dir / "contention_log.txt", args.bucket_size)
    rerr_rows, rerr_counts = parse_rerr_debug(log_dir / "rerr_debug.txt", args.bucket_size)
    precursor_rows, precursor_counts = parse_rerr_precursors(log_dir / "rerr_precursor_log.txt", args.bucket_size)
    result_metrics = parse_result(log_dir / "result.txt")

    timeseries_rows = build_timeseries(
        args.bucket_size,
        control_counts,
        contention_counts,
        contention_metrics,
        rerr_counts,
        precursor_counts,
    )

    write_csv(out_dir / "aodv_control_events.csv", sorted({key for row in control_rows for key in row.keys()}), control_rows)
    write_csv(out_dir / "contention_events.csv", sorted({key for row in contention_rows for key in row.keys()}), contention_rows)
    write_csv(out_dir / "rerr_debug_events.csv", ["time", "bucket", "node", "destinations"], rerr_rows)
    write_csv(out_dir / "rerr_precursor_events.csv", ["time", "bucket", "node", "precursors"], precursor_rows)
    write_csv(out_dir / "timeseries_summary.csv", list(timeseries_rows[0].keys()) if timeseries_rows else [
        "time_bucket_s", "bucket_size_s", "rreq_send", "rreq_recv", "rreq_duplicate_drop", "rrep_send", "rrep_recv",
        "rrep_timeout", "rerr_recv", "rerr_originated", "rerr_destinations_total", "rerr_precursors_total",
        "medium_busy", "medium_free", "backoff_start", "schedule_tx", "channel_granted",
        "avg_contention_time_s", "avg_wait_interval_s"
    ], timeseries_rows)

    with (out_dir / "run_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Recommended reads\n")
        handle.write("- timeseries_summary.csv: main file for time-bucket comparison\n")
        handle.write("- control_plane_events.png: RREQ duplicate drops, RREP timeouts, RERR originated\n")
        handle.write("- contention_pressure.png: medium busy and backoff pressure\n")
        handle.write("- contention_delay.png: waiting delay before channel grant\n")
        if result_metrics:
            handle.write("\nResult metrics\n")
            for key, value in result_metrics.items():
                handle.write(f"- {key}: {value}\n")

    graph_paths = try_plot(out_dir / "timeseries_summary.csv", out_dir)
    print(f"analysis written to: {out_dir}")
    if graph_paths:
        print("generated graphs:")
        for graph_path in graph_paths:
            print(f"- {graph_path}")
    else:
        print("matplotlib not available; CSV files were generated without graphs")


if __name__ == "__main__":
    main()
