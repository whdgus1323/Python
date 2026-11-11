from __future__ import annotations

from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape


ScenarioRows = list[dict[str, float | int]]
Point = tuple[float, float]

DATA: dict[str, ScenarioRows] = {
    "100/10": [{"nodeId": 0, "Legacy": 65.8857, "Proposed": 72.4429}, {"nodeId": 5, "Legacy": 68.3714, "Proposed": 63.1714}, {"nodeId": 8, "Legacy": 95.8571, "Proposed": 89.2}, {"nodeId": 9, "Legacy": 97.5857, "Proposed": 98.9714}, {"nodeId": 12, "Legacy": 25.2286, "Proposed": 80.5}, {"nodeId": 13, "Legacy": 68.0143, "Proposed": 63.7857}, {"nodeId": 16, "Legacy": 100.0, "Proposed": 100.0}, {"nodeId": 17, "Legacy": 89.2, "Proposed": 100.0}, {"nodeId": 20, "Legacy": 62.5, "Proposed": 62.5}, {"nodeId": 22, "Legacy": 100.0, "Proposed": 100.0}],
    "100/20": [{"nodeId": 2, "Legacy": 32.3571, "Proposed": 78.3}, {"nodeId": 3, "Legacy": 28.0857, "Proposed": 85.0}, {"nodeId": 5, "Legacy": 73.7286, "Proposed": 84.0286}, {"nodeId": 6, "Legacy": 19.0857, "Proposed": 60.7429}, {"nodeId": 8, "Legacy": 86.0, "Proposed": 90.5857}, {"nodeId": 9, "Legacy": 90.2143, "Proposed": 90.8143}, {"nodeId": 10, "Legacy": 8.48571, "Proposed": 53.0}, {"nodeId": 11, "Legacy": 73.3429, "Proposed": 90.9714}, {"nodeId": 14, "Legacy": 72.4429, "Proposed": 68.2714}, {"nodeId": 17, "Legacy": 35.3429, "Proposed": 61.2571}],
    "100/30": [{"nodeId": 2, "Legacy": 97.4, "Proposed": 97.4}, {"nodeId": 3, "Legacy": 95.0143, "Proposed": 98.6714}, {"nodeId": 5, "Legacy": 74.0, "Proposed": 86.9286}, {"nodeId": 6, "Legacy": 98.4857, "Proposed": 99.9857}, {"nodeId": 9, "Legacy": 82.8571, "Proposed": 84.5857}, {"nodeId": 11, "Legacy": 61.5857, "Proposed": 63.7143}, {"nodeId": 13, "Legacy": 41.7571, "Proposed": 50.3571}, {"nodeId": 18, "Legacy": 81.2714, "Proposed": 98.5714}, {"nodeId": 19, "Legacy": 48.5571, "Proposed": 76.6429}, {"nodeId": 22, "Legacy": 48.9857, "Proposed": 80.9286}],
    "150/10": [{"nodeId": 0, "Legacy": 65.8857, "Proposed": 72.4429}, {"nodeId": 5, "Legacy": 94.1143, "Proposed": 94.1143}, {"nodeId": 8, "Legacy": 42.8571, "Proposed": 75.3286}, {"nodeId": 9, "Legacy": 96.1857, "Proposed": 72.5143}, {"nodeId": 12, "Legacy": 21.5571, "Proposed": 75.7143}, {"nodeId": 13, "Legacy": 20.8429, "Proposed": 51.4429}, {"nodeId": 16, "Legacy": 100.0, "Proposed": 100.0}, {"nodeId": 17, "Legacy": 100.0, "Proposed": 100.0}, {"nodeId": 20, "Legacy": 61.5857, "Proposed": 81.7429}, {"nodeId": 22, "Legacy": 100.0, "Proposed": 100.0}],
    "150/20": [{"nodeId": 2, "Legacy": 44.6714, "Proposed": 74.2857}, {"nodeId": 5, "Legacy": 83.4714, "Proposed": 73.7286}, {"nodeId": 6, "Legacy": 28.1714, "Proposed": 87.3286}, {"nodeId": 8, "Legacy": 19.5, "Proposed": 85.9143}, {"nodeId": 9, "Legacy": 88.8571, "Proposed": 86.5429}, {"nodeId": 11, "Legacy": 59.2143, "Proposed": 67.7429}, {"nodeId": 13, "Legacy": 61.4, "Proposed": 64.1286}, {"nodeId": 15, "Legacy": 14.7571, "Proposed": 56.6143}, {"nodeId": 17, "Legacy": 31.5857, "Proposed": 73.1714}, {"nodeId": 18, "Legacy": 43.9286, "Proposed": 67.5143}],
    "150/30": [{"nodeId": 0, "Legacy": 24.4286, "Proposed": 65.3429}, {"nodeId": 2, "Legacy": 78.0857, "Proposed": 97.6429}, {"nodeId": 3, "Legacy": 99.0143, "Proposed": 99.0143}, {"nodeId": 5, "Legacy": 71.6143, "Proposed": 81.4714}, {"nodeId": 6, "Legacy": 86.8571, "Proposed": 96.5286}, {"nodeId": 7, "Legacy": 19.4, "Proposed": 82.3571}, {"nodeId": 8, "Legacy": 62.9143, "Proposed": 59.7}, {"nodeId": 9, "Legacy": 70.9143, "Proposed": 72.7714}, {"nodeId": 10, "Legacy": 56.0429, "Proposed": 63.6429}, {"nodeId": 11, "Legacy": 70.7286, "Proposed": 83.3857}],
    "200/10": [{"nodeId": 0, "Legacy": 65.8857, "Proposed": 72.4429}, {"nodeId": 5, "Legacy": 94.1143, "Proposed": 94.1143}, {"nodeId": 8, "Legacy": 42.8571, "Proposed": 75.3286}, {"nodeId": 9, "Legacy": 96.1857, "Proposed": 72.5143}, {"nodeId": 12, "Legacy": 21.5571, "Proposed": 75.7143}, {"nodeId": 13, "Legacy": 20.8429, "Proposed": 51.4429}, {"nodeId": 16, "Legacy": 100.0, "Proposed": 100.0}, {"nodeId": 17, "Legacy": 100.0, "Proposed": 100.0}, {"nodeId": 20, "Legacy": 61.5857, "Proposed": 81.7429}, {"nodeId": 22, "Legacy": 100.0, "Proposed": 100.0}],
    "200/20": [{"nodeId": 2, "Legacy": 44.6714, "Proposed": 74.2857}, {"nodeId": 5, "Legacy": 83.4714, "Proposed": 73.7286}, {"nodeId": 6, "Legacy": 28.1714, "Proposed": 87.3286}, {"nodeId": 8, "Legacy": 19.5, "Proposed": 85.9143}, {"nodeId": 9, "Legacy": 88.8571, "Proposed": 86.5429}, {"nodeId": 11, "Legacy": 59.2143, "Proposed": 67.7429}, {"nodeId": 13, "Legacy": 61.4, "Proposed": 64.1286}, {"nodeId": 15, "Legacy": 14.7571, "Proposed": 56.6143}, {"nodeId": 17, "Legacy": 31.5857, "Proposed": 73.1714}, {"nodeId": 18, "Legacy": 43.9286, "Proposed": 67.5143}],
    "200/30": [],
}

SCENARIO_ORDER = ["100/10", "100/20", "100/30", "150/10", "150/20", "150/30", "200/10", "200/20", "200/30"]
BLUE = "#1f77b4"
RED = "#d62728"


def line_points(values: list[float], width: float, height: float, left: float, top: float) -> list[Point]:
    if len(values) == 1:
        return [(left + width / 2, top + height - values[0] / 100 * height)]
    step = width / (len(values) - 1)
    return [(left + index * step, top + height - value / 100 * height) for index, value in enumerate(values)]


def polyline(points: list[Point], color: str) -> str:
    joined = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}" />' for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{joined}" />{circles}'


def scenario_svg_block(name: str, rows: ScenarioRows, x: float, y: float, width: float, height: float) -> str:
    title = f'<text x="{x + width / 2:.1f}" y="{y + 24:.1f}" text-anchor="middle" font-size="18" font-weight="700">{escape(name)}</text>'
    if not rows:
        return (
            f'<g><rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="white" stroke="#d0d7de" />'
            f"{title}<text x=\"{x + width / 2:.1f}\" y=\"{y + height / 2:.1f}\" text-anchor=\"middle\" "
            "font-size=\"16\" fill=\"#6b7280\">No data</text></g>"
        )

    sorted_rows = sorted(rows, key=lambda row: int(row["nodeId"]))
    node_ids = [str(int(row["nodeId"])) for row in sorted_rows]
    legacy = [float(row["Legacy"]) for row in sorted_rows]
    proposed = [float(row["Proposed"]) for row in sorted_rows]
    left, right, top, bottom = x + 52, x + width - 18, y + 42, y + height - 48
    plot_width, plot_height = right - left, bottom - top

    y_grid = "".join(
        f'<line x1="{left}" y1="{top + plot_height * step / 4:.1f}" x2="{right}" y2="{top + plot_height * step / 4:.1f}" '
        'stroke="#e5e7eb" stroke-dasharray="4 4" />'
        for step in range(5)
    )
    y_labels = "".join(
        f'<text x="{left - 10:.1f}" y="{top + plot_height * step / 4 + 4:.1f}" text-anchor="end" font-size="10" fill="#4b5563">{100 - 25 * step}</text>'
        for step in range(5)
    )
    x_labels = "".join(
        f'<text x="{left + plot_width * idx / max(len(node_ids) - 1, 1):.1f}" y="{bottom + 18:.1f}" text-anchor="middle" font-size="10" fill="#4b5563">{label}</text>'
        for idx, label in enumerate(node_ids)
    )

    legacy_points = line_points(legacy, plot_width, plot_height, left, top)
    proposed_points = line_points(proposed, plot_width, plot_height, left, top)
    avg_legacy = mean(legacy)
    avg_proposed = mean(proposed)
    delta = avg_proposed - avg_legacy
    note = (
        f'<rect x="{x + 10:.1f}" y="{y + height - 72:.1f}" width="108" height="54" rx="8" fill="#ffffff" stroke="#d1d5db" />'
        f'<text x="{x + 18:.1f}" y="{y + height - 52:.1f}" font-size="10">AVG L: {avg_legacy:.2f}</text>'
        f'<text x="{x + 18:.1f}" y="{y + height - 38:.1f}" font-size="10">AVG P: {avg_proposed:.2f}</text>'
        f'<text x="{x + 18:.1f}" y="{y + height - 24:.1f}" font-size="10">Δ: {delta:+.2f}</text>'
    )

    return (
        f'<g><rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="white" stroke="#d0d7de" />'
        f"{title}{y_grid}{y_labels}{x_labels}"
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827" />'
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#111827" />'
        f"{polyline(legacy_points, BLUE)}{polyline(proposed_points, RED)}{note}</g>"
    )


def build_grid_svg() -> str:
    width, height = 1800, 1400
    cell_w, cell_h = 540, 390
    x_positions = [40, 630, 1220]
    y_positions = [70, 500, 930]
    blocks = []
    for index, scenario in enumerate(SCENARIO_ORDER):
        row, col = divmod(index, 3)
        blocks.append(scenario_svg_block(scenario, DATA[scenario], x_positions[col], y_positions[row], cell_w, cell_h))

    legend = (
        '<g><line x1="1330" y1="28" x2="1375" y2="28" stroke="#1f77b4" stroke-width="3" />'
        '<circle cx="1352" cy="28" r="4" fill="#1f77b4" /><text x="1385" y="32" font-size="14">Legacy</text>'
        '<line x1="1460" y1="28" x2="1505" y2="28" stroke="#d62728" stroke-width="3" />'
        '<circle cx="1482" cy="28" r="4" fill="#d62728" /><text x="1515" y="32" font-size="14">Proposed</text></g>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8fafc" />'
        '<text x="40" y="34" font-size="26" font-weight="700">PDR Comparison by Scenario</text>'
        f"{legend}{''.join(blocks)}</svg>"
    )


def build_average_svg() -> str:
    summary = []
    for scenario in SCENARIO_ORDER:
        rows = DATA[scenario]
        if not rows:
            continue
        legacy = mean(float(row["Legacy"]) for row in rows)
        proposed = mean(float(row["Proposed"]) for row in rows)
        summary.append((scenario, legacy, proposed, proposed - legacy))

    width, height = 1400, 700
    left, top, right, bottom = 100, 80, 1320, 590
    plot_w, plot_h = right - left, bottom - top
    slot = plot_w / len(summary)
    bar_w = slot * 0.28
    bars = []
    labels = []
    for idx, (scenario, legacy, proposed, delta) in enumerate(summary):
        center = left + slot * idx + slot / 2
        legacy_h = plot_h * legacy / 100
        proposed_h = plot_h * proposed / 100
        bars.append(f'<rect x="{center - bar_w - 6:.1f}" y="{bottom - legacy_h:.1f}" width="{bar_w:.1f}" height="{legacy_h:.1f}" fill="{BLUE}" />')
        bars.append(f'<rect x="{center + 6:.1f}" y="{bottom - proposed_h:.1f}" width="{bar_w:.1f}" height="{proposed_h:.1f}" fill="{RED}" />')
        labels.append(f'<text x="{center:.1f}" y="{bottom + 24:.1f}" text-anchor="middle" font-size="11">{scenario}</text>')
        labels.append(f'<text x="{center:.1f}" y="{min(bottom - max(legacy_h, proposed_h) - 10, bottom - 8):.1f}" text-anchor="middle" font-size="10">{delta:+.2f}</text>')

    y_grid = "".join(
        f'<line x1="{left}" y1="{top + plot_h * step / 4:.1f}" x2="{right}" y2="{top + plot_h * step / 4:.1f}" stroke="#e5e7eb" stroke-dasharray="4 4" />'
        for step in range(5)
    )
    y_labels = "".join(
        f'<text x="{left - 10:.1f}" y="{top + plot_h * step / 4 + 4:.1f}" text-anchor="end" font-size="11">{100 - 25 * step}</text>'
        for step in range(5)
    )
    legend = (
        '<g><rect x="1080" y="24" width="18" height="18" fill="#1f77b4" /><text x="1108" y="38" font-size="14">Legacy AVG</text>'
        '<rect x="1220" y="24" width="18" height="18" fill="#d62728" /><text x="1248" y="38" font-size="14">Proposed AVG</text></g>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff" />'
        '<text x="48" y="38" font-size="26" font-weight="700">Average PDR by Scenario</text>'
        f"{legend}{y_grid}{y_labels}"
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#111827" />'
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#111827" />'
        f"{''.join(bars)}{''.join(labels)}</svg>"
    )


def main() -> None:
    output_dir = Path.cwd() / "pdr_graphs"
    output_dir.mkdir(exist_ok=True)
    scenario_path = output_dir / "pdr_scenario_grid.svg"
    average_path = output_dir / "pdr_average_summary.svg"
    scenario_path.write_text(build_grid_svg(), encoding="utf-8")
    average_path.write_text(build_average_svg(), encoding="utf-8")
    print(f"Saved: {scenario_path}")
    print(f"Saved: {average_path}")


if __name__ == "__main__":
    main()
