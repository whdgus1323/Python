#!/usr/bin/env python3
"""Generate SUMO maps by recombining spatial blocks from a baseline net.xml."""

from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BASELINE = r"C:\Users\Choe JongHyeon\Desktop\OSM_project\baseline\map.net.xml"


def fmt(value: float) -> str:
    """Format a SUMO coordinate."""
    return f"{value:.2f}"


def parse_shape(shape: str) -> list[tuple[float, float]]:
    """Parse a SUMO shape string."""
    points: list[tuple[float, float]] = []
    for token in shape.split():
        x_str, y_str = token.split(",", 1)
        points.append((float(x_str), float(y_str)))
    return points


def shape_text(points: list[tuple[float, float]]) -> str:
    """Serialize a SUMO shape string."""
    return " ".join(f"{fmt(x)},{fmt(y)}" for x, y in points)


def parse_boundary(value: str) -> tuple[float, float, float, float]:
    """Parse a SUMO boundary string."""
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid boundary: {value}")
    return parts[0], parts[1], parts[2], parts[3]


def read_baseline(path: str | Path) -> dict[str, Any]:
    """Read baseline priority junctions, road edges, and type definitions."""
    root = ET.parse(path).getroot()
    location = root.find("location")
    if location is None:
        raise ValueError("Baseline net.xml is missing <location>")
    conv = parse_boundary(location.get("convBoundary", "0,0,1,1"))

    junctions = {
        j.get("id"): {
            "id": j.get("id"),
            "x": float(j.get("x", "0")),
            "y": float(j.get("y", "0")),
        }
        for j in root.findall("junction")
        if j.get("type") == "priority" and j.get("id")
    }
    edges = []
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        from_id = edge.get("from")
        to_id = edge.get("to")
        if from_id not in junctions or to_id not in junctions:
            continue
        lane = edge.find("lane")
        if lane is None:
            continue
        shape = edge.get("shape") or lane.get("shape")
        if not shape:
            continue
        edges.append(
            {
                "id": edge.get("id", f"e{len(edges)}"),
                "from": from_id,
                "to": to_id,
                "type": edge.get("type") or "highway.residential",
                "priority": edge.get("priority") or "4",
                "speed": lane.get("speed") or "13.89",
                "shape": parse_shape(shape),
            }
        )

    types = [dict(t.attrib) for t in root.findall("type")]
    if not any(t.get("id") == "highway.residential" for t in types):
        types.append(
            {
                "id": "highway.residential",
                "priority": "4",
                "numLanes": "1",
                "speed": "13.89",
                "disallow": "tram rail_urban rail rail_electric ship",
                "oneway": "0",
            }
        )
    return {"location": dict(location.attrib), "conv": conv, "junctions": junctions, "edges": edges, "types": types}


def tile_for_point(x: float, y: float, conv: tuple[float, float, float, float], rows: int, cols: int) -> tuple[int, int]:
    """Return tile index for a coordinate."""
    min_x, min_y, max_x, max_y = conv
    col = min(cols - 1, max(0, int((x - min_x) / max(max_x - min_x, 1.0) * cols)))
    row = min(rows - 1, max(0, int((y - min_y) / max(max_y - min_y, 1.0) * rows)))
    return row, col


def tile_bounds(conv: tuple[float, float, float, float], rows: int, cols: int, tile: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return tile rectangle bounds."""
    min_x, min_y, max_x, max_y = conv
    row, col = tile
    w = (max_x - min_x) / cols
    h = (max_y - min_y) / rows
    return min_x + col * w, min_y + row * h, min_x + (col + 1) * w, min_y + (row + 1) * h


def transform_point(
    point: tuple[float, float],
    src_bounds: tuple[float, float, float, float],
    dst_bounds: tuple[float, float, float, float],
    margin: float = 5.0,
) -> tuple[float, float]:
    """Map a point from source tile coordinates into destination tile coordinates."""
    sx1, sy1, sx2, sy2 = src_bounds
    dx1, dy1, dx2, dy2 = dst_bounds
    nx = (point[0] - sx1) / max(sx2 - sx1, 1.0)
    ny = (point[1] - sy1) / max(sy2 - sy1, 1.0)
    nx = min(max(nx, 0.0), 1.0)
    ny = min(max(ny, 0.0), 1.0)
    return dx1 + margin + nx * max(dx2 - dx1 - 2 * margin, 1.0), dy1 + margin + ny * max(dy2 - dy1 - 2 * margin, 1.0)


def build_block_map(
    baseline: dict[str, Any],
    seed: int,
    rows: int = 4,
    cols: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a baseline-like map by applying small per-block deformations.

    Earlier versions physically shuffled distant blocks. That produced maps
    that were technically valid but visually too different from the baseline.
    This version keeps the baseline topology and block positions, then applies
    mild block-level warp so generated maps remain recognizably baseline-like.
    """
    rng = random.Random(seed)
    conv = baseline["conv"]
    all_tiles = [(r, c) for r in range(rows) for c in range(cols)]
    tile_map = {tile: tile for tile in all_tiles}
    min_x, min_y, max_x, max_y = conv

    tile_params: dict[tuple[int, int], dict[str, float]] = {}
    for tile in all_tiles:
        bounds = tile_bounds(conv, rows, cols, tile)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        tile_params[tile] = {
            "dx": rng.uniform(-0.055, 0.055) * width,
            "dy": rng.uniform(-0.055, 0.055) * height,
            "angle": math.radians(rng.uniform(-6.0, 6.0)),
            "scale": rng.uniform(0.94, 1.06),
        }

    def warp_point(point: tuple[float, float], node_key: str = "") -> tuple[float, float]:
        tile = tile_for_point(point[0], point[1], conv, rows, cols)
        bounds = tile_bounds(conv, rows, cols, tile)
        cx = (bounds[0] + bounds[2]) / 2.0
        cy = (bounds[1] + bounds[3]) / 2.0
        params = tile_params[tile]
        x = point[0] - cx
        y = point[1] - cy
        ca = math.cos(params["angle"])
        sa = math.sin(params["angle"])
        sx = params["scale"] * (x * ca - y * sa)
        sy = params["scale"] * (x * sa + y * ca)
        local_rng = random.Random(f"{seed}:{node_key}")
        jitter_x = local_rng.uniform(-2.0, 2.0) if node_key else 0.0
        jitter_y = local_rng.uniform(-2.0, 2.0) if node_key else 0.0
        nx = min(max(cx + sx + params["dx"] + jitter_x, min_x), max_x)
        ny = min(max(cy + sy + params["dy"] + jitter_y, min_y), max_y)
        return nx, ny

    source_node_tile: dict[str, tuple[int, int]] = {}
    nodes_by_source_tile: dict[tuple[int, int], list[str]] = defaultdict(list)
    for node_id, node in baseline["junctions"].items():
        tile = tile_for_point(node["x"], node["y"], conv, rows, cols)
        source_node_tile[node_id] = tile
        nodes_by_source_tile[tile].append(node_id)

    generated_nodes: list[dict[str, Any]] = []
    node_id_map: dict[str, str] = {}
    target_tile_nodes: dict[tuple[int, int], list[str]] = defaultdict(list)
    generated_node_lookup: dict[str, dict[str, Any]] = {}

    for dst_tile, src_tile in tile_map.items():
        for old_id in nodes_by_source_tile[src_tile]:
            old = baseline["junctions"][old_id]
            new_id = f"n_{dst_tile[0]}_{dst_tile[1]}_{old_id}"
            x, y = warp_point((old["x"], old["y"]), old_id)
            node = {"id": new_id, "x": x, "y": y, "type": "priority", "target_tile": dst_tile}
            node_id_map[old_id] = new_id
            generated_nodes.append(node)
            generated_node_lookup[new_id] = node
            target_tile_nodes[dst_tile].append(new_id)

    generated_edges: list[dict[str, Any]] = []
    used_edge_ids: set[str] = set()

    def add_edge(edge_id: str, from_id: str, to_id: str, shape: list[tuple[float, float]] | None = None, edge_type: str = "highway.residential") -> None:
        if from_id == to_id or from_id not in generated_node_lookup or to_id not in generated_node_lookup:
            return
        if edge_id in used_edge_ids:
            return
        used_edge_ids.add(edge_id)
        a = generated_node_lookup[from_id]
        b = generated_node_lookup[to_id]
        if shape is None:
            shape = [(a["x"], a["y"]), (b["x"], b["y"])]
        generated_edges.append(
            {
                "id": edge_id,
                "from": from_id,
                "to": to_id,
                "type": edge_type,
                "priority": "4",
                "numLanes": "1",
                "speed": "13.89",
                "shape": shape,
            }
        )

    for edge in baseline["edges"]:
        from_tile = source_node_tile.get(edge["from"])
        to_tile = source_node_tile.get(edge["to"])
        if from_tile is None or to_tile is None:
            continue
        if edge["from"] not in node_id_map or edge["to"] not in node_id_map:
            continue
        dst_tile = next(dst for dst, src in tile_map.items() if src == from_tile)
        safe_id = "".join(ch if ch.isalnum() else "_" for ch in edge["id"])
        # Do not reuse baseline lane/edge shapes here. Those shapes already
        # contain SUMO lane offsets; feeding them back into netconvert makes
        # opposite directions render as separated double roads. Let netconvert
        # derive clean lane geometry from the warped junction coordinates.
        add_edge(f"b_{dst_tile[0]}_{dst_tile[1]}_{safe_id}", node_id_map[edge["from"]], node_id_map[edge["to"]], None, edge["type"])

    def boundary_nodes(tile: tuple[int, int], side: str) -> list[str]:
        bounds = tile_bounds(conv, rows, cols, tile)
        x1, y1, x2, y2 = bounds
        nodes = target_tile_nodes[tile]
        if not nodes:
            return []
        if side == "right":
            return sorted(nodes, key=lambda nid: abs(generated_node_lookup[nid]["x"] - x2))[:4]
        if side == "left":
            return sorted(nodes, key=lambda nid: abs(generated_node_lookup[nid]["x"] - x1))[:4]
        if side == "top":
            return sorted(nodes, key=lambda nid: abs(generated_node_lookup[nid]["y"] - y2))[:4]
        return sorted(nodes, key=lambda nid: abs(generated_node_lookup[nid]["y"] - y1))[:4]

    connect_components(generated_nodes, generated_edges, generated_node_lookup)
    repair_low_degree_nodes(generated_nodes, generated_edges, generated_node_lookup)
    return generated_nodes, generated_edges


def repair_low_degree_nodes(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], lookup: dict[str, dict[str, Any]], min_degree: int = 2) -> None:
    """Connect low-degree nodes to nearby nodes to avoid SUMO dead-end dots."""
    def degrees() -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {n["id"]: set() for n in nodes}
        for edge in edges:
            adj[edge["from"]].add(edge["to"])
            adj[edge["to"]].add(edge["from"])
        return adj

    edge_index = 0
    for _ in range(8):
        adj = degrees()
        low = [node_id for node_id, neighbors in adj.items() if len(neighbors) < min_degree]
        if not low:
            return
        changed = False
        for node_id in low:
            node = lookup[node_id]
            candidates = []
            for other_id, other in lookup.items():
                if other_id == node_id or other_id in adj[node_id]:
                    continue
                dist = math.hypot(node["x"] - other["x"], node["y"] - other["y"])
                candidates.append((dist, other_id))
            candidates.sort()
            for _dist, other_id in candidates[:1]:
                for from_id, to_id in ((node_id, other_id), (other_id, node_id)):
                    edge_index += 1
                    a = lookup[from_id]
                    b = lookup[to_id]
                    edges.append(
                        {
                            "id": f"repair_{edge_index}",
                            "from": from_id,
                            "to": to_id,
                            "type": "highway.residential",
                            "priority": "4",
                            "numLanes": "1",
                            "speed": "13.89",
                            "shape": [(a["x"], a["y"]), (b["x"], b["y"])],
                        }
                    )
                changed = True
                break
        if not changed:
            return


def connect_components(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], lookup: dict[str, dict[str, Any]]) -> None:
    """Add nearest bidirectional links until the plain graph is connected."""
    def components() -> list[set[str]]:
        adj: dict[str, set[str]] = {n["id"]: set() for n in nodes}
        for e in edges:
            adj[e["from"]].add(e["to"])
            adj[e["to"]].add(e["from"])
        unseen = set(adj)
        comps = []
        while unseen:
            start = unseen.pop()
            comp = {start}
            stack = [start]
            while stack:
                cur = stack.pop()
                for nxt in adj[cur]:
                    if nxt in unseen:
                        unseen.remove(nxt)
                        comp.add(nxt)
                        stack.append(nxt)
            comps.append(comp)
        return comps

    edge_index = 0
    while True:
        comps = components()
        if len(comps) <= 1:
            return
        base = comps[0]
        other = comps[1]
        best = None
        for a in base:
            for b in other:
                na = lookup[a]
                nb = lookup[b]
                dist = math.hypot(na["x"] - nb["x"], na["y"] - nb["y"])
                if best is None or dist < best[0]:
                    best = (dist, a, b)
        if best is None:
            return
        _dist, a, b = best
        for from_id, to_id in ((a, b), (b, a)):
            edge_index += 1
            na = lookup[from_id]
            nb = lookup[to_id]
            edges.append(
                {
                    "id": f"component_{edge_index}",
                    "from": from_id,
                    "to": to_id,
                    "type": "highway.residential",
                    "priority": "4",
                    "numLanes": "1",
                    "speed": "13.89",
                    "shape": [(na["x"], na["y"]), (nb["x"], nb["y"])],
                }
            )


def write_plain_files(folder: Path, baseline: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    """Write SUMO plain nod/edg/typ files."""
    folder.mkdir(parents=True, exist_ok=True)
    nod = folder / "plain.nod.xml"
    edg = folder / "plain.edg.xml"
    typ = folder / "plain.typ.xml"

    node_root = ET.Element("nodes")
    for node in nodes:
        ET.SubElement(node_root, "node", {"id": node["id"], "x": fmt(node["x"]), "y": fmt(node["y"]), "type": "priority"})
    ET.indent(node_root, space="    ")
    ET.ElementTree(node_root).write(nod, encoding="UTF-8", xml_declaration=True)

    edge_root = ET.Element("edges")
    for edge in edges:
        attrs = {
            "id": edge["id"],
            "from": edge["from"],
            "to": edge["to"],
            "priority": edge["priority"],
            "type": edge["type"],
            "numLanes": edge["numLanes"],
            "speed": edge["speed"],
        }
        if edge.get("shape"):
            attrs["shape"] = shape_text(edge["shape"])
        ET.SubElement(edge_root, "edge", attrs)
    ET.indent(edge_root, space="    ")
    ET.ElementTree(edge_root).write(edg, encoding="UTF-8", xml_declaration=True)

    type_root = ET.Element("types")
    seen = set()
    for type_attrs in baseline["types"]:
        if not type_attrs.get("id") or type_attrs["id"] in seen:
            continue
        seen.add(type_attrs["id"])
        attrs = dict(type_attrs)
        if attrs["id"] in {"highway.residential", "highway.tertiary", "highway.secondary", "highway.primary"}:
            attrs["numLanes"] = "1"
        ET.SubElement(type_root, "type", attrs)
    ET.indent(type_root, space="    ")
    ET.ElementTree(type_root).write(typ, encoding="UTF-8", xml_declaration=True)
    return nod, edg, typ


def ensure_sumo_cfg(folder: Path, end: int = 200) -> None:
    """Write map.sumo.cfg if missing."""
    cfg = folder / "map.sumo.cfg"
    if cfg.exists():
        return
    cfg.write_text(
        f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>

<configuration>
  <input>
    <net-file value=\"map.net.xml\"/>
    <route-files value=\"map.rou.xml\"/>
  </input>
  <time>
    <begin value=\"0\"/>
    <end value=\"{end}\"/>
  </time>
  <processing>
    <max-num-vehicles value=\"100\"/>
  </processing>
  <output>
    <fcd-output value=\"map.xml\"/>
  </output>
</configuration>
""",
        encoding="UTF-8",
    )


def run(cmd: list[str], cwd: Path) -> None:
    """Run a command with a useful error."""
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed\n{completed.stdout}\n{completed.stderr}")


def validate_with_existing(output: Path) -> dict[str, Any]:
    """Validate with the existing synthetic generator's final network inspector."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_sumo_net import inspect_final_network

    return inspect_final_network(output, reject_straight_degree_two=False, reject_large_blank_areas=True)


def generate_one(baseline_path: Path, output_dir: Path, seed: int, run_sumo: bool) -> dict[str, Any]:
    """Generate one block-combined SUMO map."""
    baseline = read_baseline(baseline_path)
    nodes, edges = build_block_map(baseline, seed)
    with tempfile.TemporaryDirectory(prefix="sumo_blocks_") as tmp_name:
        tmp = Path(tmp_name)
        nod, edg, typ = write_plain_files(tmp, baseline, nodes, edges)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_net = output_dir / "map.net.xml"
        run(["netconvert", "-n", str(nod), "-e", str(edg), "-t", str(typ), "-o", str(out_net), "--no-warnings"], output_dir)
    ensure_sumo_cfg(output_dir)
    stats = validate_with_existing(out_net)
    if run_sumo:
        python_exe = shutil.which("python") or sys.executable
        random_trips = shutil.which("randomTrips.py") or "randomTrips.py"
        run([python_exe, random_trips, "-n", "map.net.xml", "-o", "map.trips.xml", "-r", "map.rou.xml", "-p", "0.3", "-e", "200", "--seed", str(seed)], output_dir)
        run(["duarouter", "-n", "map.net.xml", "--route-files", "map.trips.xml", "-o", "map.rou.xml", "--ignore-errors"], output_dir)
        run(["sumo", "-c", "map.sumo.cfg", "--fcd-output", "map.xml", "--end", "200"], output_dir)
    return stats


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate SUMO maps by recombining baseline road blocks.")
    parser.add_argument("--input", default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", default=r"C:\Users\Choe JongHyeon\Desktop\OSM_project\block_map")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-count", type=int, default=0)
    parser.add_argument("--batch-start-index", type=int, default=1)
    parser.add_argument("--run-sumo", action="store_true")
    args = parser.parse_args()

    try:
        baseline_path = Path(args.input)
        if args.batch_count:
            root = Path(args.output_dir)
            for index in range(args.batch_start_index, args.batch_start_index + args.batch_count):
                out = root / f"map_{index}"
                stats = generate_one(baseline_path, out, index, args.run_sumo)
                print(f"map_{index}: priority={stats['priority_junction_count']} edges={stats['normal_edge_count']} lanes={dict(stats['normal_edge_lane_distribution'])}")
        else:
            stats = generate_one(baseline_path, Path(args.output_dir), args.seed, args.run_sumo)
            print(f"Done: priority={stats['priority_junction_count']} edges={stats['normal_edge_count']} lanes={dict(stats['normal_edge_lane_distribution'])}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
