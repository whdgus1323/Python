from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_BASE = Path(r"C:\Users\Choe JongHyeon\Desktop\map")


@dataclass(frozen=True)
class RunStatus:
    low: int
    high: int
    n: int
    path: Path
    status: str
    detail: str


def parse_num(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix):]
    return int(tail) if tail.isdigit() else None


def iter_pair_dirs(base: Path) -> Iterable[tuple[int, int, Path]]:
    for low_dir in sorted(base.iterdir(), key=lambda p: p.name):
        if not low_dir.is_dir():
            continue
        low = parse_num(low_dir.name, "m_")
        if low is None:
            continue
        for high_dir in sorted(low_dir.iterdir(), key=lambda p: p.name):
            if not high_dir.is_dir():
                continue
            high = parse_num(high_dir.name, "p_")
            if high is None:
                continue
            yield low, high, high_dir


def classify_run(run_dir: Path) -> tuple[str, str]:
    result_file = run_dir / "result.txt"
    if result_file.exists():
        try:
            text = result_file.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return "partial", f"result.txt unreadable: {exc}"
        if "PDR" in text and "sent:" in text and "received:" in text:
            return "complete", "result.txt contains final summary"
        return "partial", "result.txt exists but final summary is missing"

    files = [p for p in run_dir.iterdir() if p.is_file()]
    if files:
        return "partial", "files exist but result.txt is missing"
    return "pending", "no output files"


def collect_statuses(base: Path) -> list[RunStatus]:
    statuses: list[RunStatus] = []
    for low, high, pair_dir in iter_pair_dirs(base):
        run_dirs = [p for p in pair_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        for run_dir in sorted(run_dirs, key=lambda p: int(p.name)):
            status, detail = classify_run(run_dir)
            statuses.append(
                RunStatus(
                    low=low,
                    high=high,
                    n=int(run_dir.name),
                    path=run_dir,
                    status=status,
                    detail=detail,
                )
            )
    return statuses


def write_csv(statuses: list[RunStatus], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["low", "high", "n", "status", "detail", "path"])
        for item in statuses:
            writer.writerow([item.low, item.high, item.n, item.status, item.detail, str(item.path)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan CBR sweep result folders and tell you where to resume."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Base folder that contains m_*/p_*/n directories. Default: {DEFAULT_BASE}",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV export path for the scan result.",
    )
    args = parser.parse_args()

    base = args.base
    if not base.exists():
        raise SystemExit(f"Base path does not exist: {base}")

    statuses = collect_statuses(base)
    if not statuses:
        raise SystemExit(f"No m_*/p_*/n folders found under: {base}")

    complete = [s for s in statuses if s.status == "complete"]
    partial = [s for s in statuses if s.status == "partial"]
    pending = [s for s in statuses if s.status == "pending"]

    print(f"Base: {base}")
    print(f"Total runs found: {len(statuses)}")
    print(f"Complete: {len(complete)}")
    print(f"Partial/crashed: {len(partial)}")
    print(f"Pending: {len(pending)}")
    print()

    if partial:
        first_partial = min(partial, key=lambda s: (s.low, s.high, s.n))
        print("First partial run:")
        print(f"  low={first_partial.low}, high={first_partial.high}, n={first_partial.n}")
        print(f"  path={first_partial.path}")
        print(f"  detail={first_partial.detail}")
        print()

    if pending:
        first_pending = min(pending, key=lambda s: (s.low, s.high, s.n))
        print("First pending run:")
        print(f"  low={first_pending.low}, high={first_pending.high}, n={first_pending.n}")
        print(f"  path={first_pending.path}")
        print()

    next_target = None
    if partial:
        next_target = min(partial, key=lambda s: (s.low, s.high, s.n))
        print("Suggested restart target:")
        print("  Retry the first partial run before moving on.")
    elif pending:
        next_target = min(pending, key=lambda s: (s.low, s.high, s.n))
        print("Suggested restart target:")
        print("  Start from the first pending run.")
    else:
        print("All discovered runs are complete.")

    if next_target is not None:
        print(f"  low={next_target.low}, high={next_target.high}, n={next_target.n}")
        print(f"  path={next_target.path}")
        print()

    if partial:
        print("Partial runs:")
        for item in sorted(partial, key=lambda s: (s.low, s.high, s.n))[:30]:
            print(f"  m_{item.low}/p_{item.high}/{item.n} -> {item.detail}")
        if len(partial) > 30:
            print(f"  ... {len(partial) - 30} more")
        print()

    if args.csv is not None:
        write_csv(statuses, args.csv)
        print(f"CSV written to: {args.csv}")


if __name__ == "__main__":
    main()
