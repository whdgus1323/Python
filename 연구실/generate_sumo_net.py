#!/usr/bin/env python3
"""Generate a synthetic SUMO net.xml file from a reference network profile.

The generator parses an existing SUMO ``map.net.xml`` file, copies its
``location`` metadata and ``type`` definitions, then creates a new irregular
grid-like road network with priority/internal junctions, edges, lanes, and
connections.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PROJ_PARAMETER = "+proj=utm +zone=52 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
DEFAULT_REQUIRED_TYPES = {
    "highway.residential": {
        "id": "highway.residential",
        "priority": "4",
        "numLanes": "1",
        "speed": "13.89",
        "disallow": "tram rail_urban rail rail_electric ship",
        "oneway": "0",
    },
    "highway.tertiary": {
        "id": "highway.tertiary",
        "priority": "6",
        "numLanes": "1",
        "speed": "22.22",
        "disallow": "tram rail_urban rail rail_electric ship",
        "oneway": "0",
    },
    "highway.secondary": {
        "id": "highway.secondary",
        "priority": "7",
        "numLanes": "2",
        "speed": "27.78",
        "disallow": "tram rail_urban rail rail_electric ship",
        "oneway": "0",
    },
    "highway.primary": {
        "id": "highway.primary",
        "priority": "9",
        "numLanes": "2",
        "speed": "27.78",
        "disallow": "tram rail_urban rail rail_electric ship",
        "oneway": "0",
    },
    "highway.service": {
        "id": "highway.service",
        "priority": "2",
        "numLanes": "1",
        "speed": "5.56",
        "allow": "delivery bicycle pedestrian",
        "oneway": "0",
    },
    "highway.unclassified": {
        "id": "highway.unclassified",
        "priority": "3",
        "numLanes": "1",
        "speed": "13.89",
        "disallow": "tram rail_urban rail rail_electric ship",
        "oneway": "0",
    },
}

ROAD_SPEEDS = {
    "highway.residential": 13.89,
    "highway.tertiary": 22.22,
    "highway.secondary": 27.78,
    "highway.primary": 27.78,
    "highway.service": 5.56,
    "highway.unclassified": 13.89,
}


def fmt(value: float) -> str:
    """Format numeric XML attributes in the style normally used by SUMO."""
    return f"{value:.2f}"


def parse_shape(shape: str) -> list[tuple[float, float]]:
    """Parse a SUMO lane or junction shape string into coordinate tuples."""
    points = []
    for token in shape.split():
        x_str, y_str = token.split(",", 1)
        points.append((float(x_str), float(y_str)))
    return points


def shape_length(points: list[tuple[float, float]]) -> float:
    """Return the polyline length for a list of coordinates."""
    return sum(
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    )


def shape_to_text(points: list[tuple[float, float]]) -> str:
    """Convert coordinate tuples to a SUMO shape attribute string."""
    return " ".join(f"{fmt(x)},{fmt(y)}" for x, y in points)


def parse_boundary(value: str) -> tuple[float, float, float, float]:
    """Parse a four-value SUMO boundary string."""
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Boundary must contain four comma-separated values: {value}")
    return parts[0], parts[1], parts[2], parts[3]


def parse_reference_net(input_path: str | Path) -> dict[str, Any]:
    """Parse a reference SUMO net.xml and extract reusable metadata/statistics."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference net.xml does not exist: {path}")

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse reference XML {path}: {exc}") from exc

    if root.tag != "net":
        raise ValueError(f"Reference XML root must be <net>, got <{root.tag}>")

    location = dict(root.find("location").attrib) if root.find("location") is not None else {}
    types = [dict(type_elem.attrib) for type_elem in root.findall("type")]
    type_by_id = {type_attr["id"]: type_attr for type_attr in types if "id" in type_attr}
    for type_id, attrs in DEFAULT_REQUIRED_TYPES.items():
        if type_id not in type_by_id:
            types.append(dict(attrs))
            type_by_id[type_id] = dict(attrs)

    edges = root.findall("edge")
    junctions = root.findall("junction")
    connections = root.findall("connection")
    lanes = [lane for edge in edges for lane in edge.findall("lane")]

    normal_edges = [edge for edge in edges if edge.get("function") != "internal"]
    internal_edges = [edge for edge in edges if edge.get("function") == "internal"]

    stats = {
        "net_version": root.get("version", "1.0"),
        "edge_count": len(edges),
        "normal_edge_count": len(normal_edges),
        "internal_edge_count": len(internal_edges),
        "lane_count": len(lanes),
        "junction_count": len(junctions),
        "priority_junction_count": sum(1 for j in junctions if j.get("type") == "priority"),
        "internal_junction_count": sum(1 for j in junctions if j.get("type") == "internal"),
        "connection_count": len(connections),
        "junction_type_counts": Counter(j.get("type", "") for j in junctions),
        "edge_type_counts": Counter(edge.get("type", "") for edge in normal_edges),
    }

    return {
        "root_attributes": dict(root.attrib),
        "location": location,
        "types": types,
        "type_by_id": type_by_id,
        "stats": stats,
    }


def adjacent_cell_pairs(cells: set[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return adjacent horizontal/vertical cell pairs for a grid graph."""
    pairs = []
    for row, col in sorted(cells):
        for neighbor in ((row, col + 1), (row + 1, col)):
            if neighbor in cells:
                pairs.append(((row, col), neighbor))
    return pairs


def graph_is_connected_and_has_no_dead_ends(
    nodes: set[Any],
    pairs: list[tuple[Any, Any]],
    min_degree: int = 2,
    positions: dict[Any, tuple[float, float]] | None = None,
    reject_straight_degree_two: bool = False,
) -> bool:
    """Check connectivity, dead ends, and optionally straight pass-through nodes."""
    if not nodes:
        return False

    adjacency: dict[Any, set[Any]] = {node: set() for node in nodes}
    for a, b in pairs:
        if a not in adjacency or b not in adjacency:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    if any(len(neighbors) < min_degree for neighbors in adjacency.values()):
        return False
    if reject_straight_degree_two and positions is not None:
        for node, neighbors in adjacency.items():
            if len(neighbors) != 2:
                continue
            n1, n2 = list(neighbors)
            x, y = positions[node]
            x1, y1 = positions[n1]
            x2, y2 = positions[n2]
            v1 = (x1 - x, y1 - y)
            v2 = (x2 - x, y2 - y)
            len1 = math.hypot(*v1)
            len2 = math.hypot(*v2)
            if len1 <= 0 or len2 <= 0:
                return False
            cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
            if cos_angle < -0.95:
                return False

    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(nodes)


def generate_junction_grid(
    width_m: float,
    height_m: float,
    rows: int,
    cols: int,
    junction_target: int,
    jitter_m: float,
    seed: int,
    layout_variation: float = 0.35,
) -> dict[str, dict[str, Any]]:
    """Generate priority junctions on an irregular grid with deterministic jitter."""
    if rows < 2 or cols < 2:
        raise ValueError("rows and cols must both be at least 2")
    if junction_target < 2:
        raise ValueError("junction_target must be at least 2")

    rng = random.Random(seed)
    all_cells = [(row, col) for row in range(rows) for col in range(cols)]
    keep_count = min(len(all_cells), junction_target)

    # Prefer removing from the full grid instead of sampling from scratch so the
    # generated network keeps the visual density of a road map. A removal is
    # accepted only if the remaining grid stays connected and has no dead ends.
    removed: set[tuple[int, int]] = set()
    candidates = all_cells[:]
    rng.shuffle(candidates)
    for row, col in candidates:
        if len(all_cells) - len(removed) <= keep_count:
            break
        # Keep corners and avoid stripping too many nodes from the boundary.
        is_corner = row in (0, rows - 1) and col in (0, cols - 1)
        if is_corner:
            continue
        trial_removed = removed | {(row, col)}
        active_trial = set(all_cells) - trial_removed
        cell_positions = {cell: (float(cell[1]), float(cell[0])) for cell in active_trial}
        if graph_is_connected_and_has_no_dead_ends(
            active_trial,
            adjacent_cell_pairs(active_trial),
            positions=cell_positions,
            reject_straight_degree_two=True,
        ):
            removed = trial_removed

    active_cells = [cell for cell in all_cells if cell not in removed]
    def axis_positions(count: int, total: float) -> list[float]:
        if count <= 1:
            return [0.0]
        if layout_variation <= 0:
            return [index * total / (count - 1) for index in range(count)]
        weights = [
            max(0.35, 1.0 + rng.uniform(-layout_variation, layout_variation))
            for _ in range(count - 1)
        ]
        scale = total / sum(weights)
        positions = [0.0]
        current = 0.0
        for weight in weights:
            current += weight * scale
            positions.append(current)
        positions[-1] = total
        return positions

    x_positions = axis_positions(cols, width_m)
    y_positions = axis_positions(rows, height_m)
    dx = width_m / max(cols - 1, 1)
    dy = height_m / max(rows - 1, 1)
    max_jitter_x = min(jitter_m, dx * 0.30)
    max_jitter_y = min(jitter_m, dy * 0.30)

    junctions: dict[str, dict[str, Any]] = {}
    for index, (row, col) in enumerate(active_cells, start=1):
        base_x = x_positions[col]
        base_y = y_positions[row]
        if col in (0, cols - 1):
            x = base_x
        else:
            x = base_x + rng.uniform(-max_jitter_x, max_jitter_x)
        if row in (0, rows - 1):
            y = base_y
        else:
            y = base_y + rng.uniform(-max_jitter_y, max_jitter_y)
        x = min(max(x, 0.0), width_m)
        y = min(max(y, 0.0), height_m)
        junction_id = f"j_{index:03d}"
        junctions[junction_id] = {"id": junction_id, "row": row, "col": col, "x": x, "y": y}

    return junctions


def choose_road_type(row: int, col: int, rows: int, cols: int, rng: random.Random) -> str:
    """Choose a plausible highway type for a grid segment."""
    center_rows = {rows // 2, max(0, rows // 2 - 1)}
    center_cols = {cols // 2, max(0, cols // 2 - 1)}
    if row in center_rows and col in center_cols:
        return "highway.primary"
    if row in center_rows or col in center_cols:
        return "highway.secondary" if rng.random() < 0.45 else "highway.tertiary"
    if rng.random() < 0.08:
        return "highway.unclassified"
    if rng.random() < 0.05:
        return "highway.service"
    return "highway.residential"


def lane_count_for_type(type_id: str, type_by_id: dict[str, dict[str, str]]) -> int:
    """Return the lane count for generated roads.

    The requested network uses two-lane roads in the visual/physical sense:
    one lane per direction. SUMO stores opposite directions as separate edges,
    so each directed normal edge receives one lane.
    """
    return 1


def select_road_pairs_without_dead_ends(
    all_pairs: list[tuple[str, str, int, int]],
    junction_ids: set[str],
    junction_positions: dict[str, tuple[float, float]],
    road_missing_prob: float,
    rng: random.Random,
) -> list[tuple[str, str, int, int]]:
    """Remove optional road segments without disconnecting or creating non-intersections."""
    selected = all_pairs[:]
    candidates = selected[:]
    rng.shuffle(candidates)

    for pair in candidates:
        if rng.random() >= road_missing_prob:
            continue
        trial = [item for item in selected if item != pair]
        graph_pairs = [(a, b) for a, b, _row, _col in trial]
        if graph_is_connected_and_has_no_dead_ends(
            junction_ids,
            graph_pairs,
            positions=junction_positions,
            reject_straight_degree_two=True,
        ):
            selected = trial

    return selected


def lane_shape_for_index(
    from_xy: tuple[float, float],
    to_xy: tuple[float, float],
    lane_index: int,
    lane_count: int,
) -> list[tuple[float, float]]:
    """Create a slightly offset lane shape for one lane of an edge."""
    x1, y1 = from_xy
    x2, y2 = to_xy
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 0:
        return [(x1, y1), (x2, y2)]
    nx = -dy / length
    ny = dx / length
    offset = (lane_index - (lane_count - 1) / 2.0) * 3.2
    return [(x1 + nx * offset, y1 + ny * offset), (x2 + nx * offset, y2 + ny * offset)]


def generate_edges(
    junctions: dict[str, dict[str, Any]],
    rows: int,
    cols: int,
    road_missing_prob: float,
    seed: int,
    type_by_id: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Generate bidirectional normal edges and their lanes between adjacent grid nodes."""
    rng = random.Random(seed + 101)
    by_cell = {(j["row"], j["col"]): jid for jid, j in junctions.items()}
    all_pairs: list[tuple[str, str, int, int]] = []

    for row in range(rows):
        for col in range(cols):
            here = by_cell.get((row, col))
            if here is None:
                continue
            for nrow, ncol in ((row, col + 1), (row + 1, col)):
                there = by_cell.get((nrow, ncol))
                if there is None:
                    continue
                all_pairs.append((here, there, row, col))

    undirected_pairs = select_road_pairs_without_dead_ends(
        all_pairs,
        set(junctions.keys()),
        {jid: (j["x"], j["y"]) for jid, j in junctions.items()},
        road_missing_prob,
        rng,
    )

    if not undirected_pairs:
        raise ValueError("No road segments were generated; lower road_missing_prob or increase grid size")

    edges: list[dict[str, Any]] = []
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_index = 1

    for from_jid, to_jid, row, col in undirected_pairs:
        road_type = choose_road_type(row, col, rows, cols, rng)
        priority = type_by_id.get(road_type, DEFAULT_REQUIRED_TYPES.get(road_type, {})).get("priority", "4")
        speed = ROAD_SPEEDS.get(road_type, float(type_by_id.get(road_type, {}).get("speed", 13.89)))
        lane_count = lane_count_for_type(road_type, type_by_id)

        for a, b, prefix in ((from_jid, to_jid, "e"), (to_jid, from_jid, "-e")):
            edge_id = f"{prefix}{edge_index:04d}"
            from_xy = (junctions[a]["x"], junctions[a]["y"])
            to_xy = (junctions[b]["x"], junctions[b]["y"])
            lanes = []
            for lane_index in range(lane_count):
                lane_points = lane_shape_for_index(from_xy, to_xy, lane_index, lane_count)
                lanes.append(
                    {
                        "id": f"{edge_id}_{lane_index}",
                        "index": str(lane_index),
                        "speed": fmt(speed),
                        "length": fmt(max(shape_length(lane_points), 0.01)),
                        "shape": shape_to_text(lane_points),
                    }
                )
            edge = {
                "id": edge_id,
                "from": a,
                "to": b,
                "priority": priority,
                "type": road_type,
                "lanes": lanes,
            }
            if "disallow" in type_by_id.get(road_type, {}):
                for lane in lanes:
                    lane["disallow"] = type_by_id[road_type]["disallow"]
            edges.append(edge)
            outgoing[a].append(edge)
            incoming[b].append(edge)
        edge_index += 1

    return edges, incoming, outgoing


def build_turn_candidates(
    junctions: dict[str, dict[str, Any]],
    incoming: dict[str, list[dict[str, Any]]],
    outgoing: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build non-U-turn candidate movements through each priority junction."""
    candidates: list[dict[str, Any]] = []
    for junction_id in junctions:
        for in_edge in incoming.get(junction_id, []):
            for out_edge in outgoing.get(junction_id, []):
                if in_edge["from"] == out_edge["to"]:
                    continue
                candidates.append({"junction": junction_id, "from": in_edge, "to": out_edge})
    return candidates


def internal_turn_shape(
    junction: dict[str, Any],
    in_edge: dict[str, Any],
    out_edge: dict[str, Any],
    junctions: dict[str, dict[str, Any]],
) -> list[tuple[float, float]]:
    """Create a short curved-looking internal lane through a junction."""
    jx, jy = junction["x"], junction["y"]
    in_from = junctions[in_edge["from"]]
    out_to = junctions[out_edge["to"]]

    def near_center(outer: dict[str, Any]) -> tuple[float, float]:
        vx = outer["x"] - jx
        vy = outer["y"] - jy
        dist = math.hypot(vx, vy) or 1.0
        return jx + vx / dist * 4.5, jy + vy / dist * 4.5

    p1 = near_center(in_from)
    p3 = near_center(out_to)
    p2 = (jx, jy)
    return [p1, p2, p3]


def generate_internal_edges(
    junctions: dict[str, dict[str, Any]],
    turn_candidates: list[dict[str, Any]],
    internal_junction_target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], str]]:
    """Generate internal junctions/edges for a subset of turning movements."""
    rng = random.Random(seed + 202)
    shuffled = turn_candidates[:]
    rng.shuffle(shuffled)
    selected = shuffled[: max(0, min(internal_junction_target, len(shuffled)))]

    internal_edges: list[dict[str, Any]] = []
    internal_junctions: list[dict[str, Any]] = []
    via_by_turn: dict[tuple[str, str], str] = {}
    per_junction_counter: Counter[str] = Counter()

    for candidate in selected:
        junction_id = candidate["junction"]
        local_index = per_junction_counter[junction_id]
        per_junction_counter[junction_id] += 1
        edge_id = f":{junction_id}_{local_index}"
        lane_id = f"{edge_id}_0"
        junction = junctions[junction_id]
        points = internal_turn_shape(junction, candidate["from"], candidate["to"], junctions)
        length = max(shape_length(points), 0.01)
        speed = min(float(candidate["from"]["lanes"][0]["speed"]), float(candidate["to"]["lanes"][0]["speed"]), 13.89)

        internal_edges.append(
            {
                "id": edge_id,
                "function": "internal",
                "lanes": [
                    {
                        "id": lane_id,
                        "index": "0",
                        "speed": fmt(speed),
                        "length": fmt(length),
                        "shape": shape_to_text(points),
                    }
                ],
            }
        )
        mid_x = sum(x for x, _ in points) / len(points)
        mid_y = sum(y for _, y in points) / len(points)
        internal_junctions.append(
            {
                "id": lane_id,
                "type": "internal",
                "x": fmt(mid_x),
                "y": fmt(mid_y),
                "incLanes": f"{candidate['from']['lanes'][0]['id']}",
                "intLanes": lane_id,
            }
        )
        via_by_turn[(candidate["from"]["id"], candidate["to"]["id"])] = lane_id

    return internal_edges, internal_junctions, via_by_turn


def direction_code(in_edge: dict[str, Any], out_edge: dict[str, Any], junctions: dict[str, dict[str, Any]]) -> str:
    """Classify a turn direction as left, right, or straight for SUMO-like metadata."""
    j = junctions[in_edge["to"]]
    a = junctions[in_edge["from"]]
    b = junctions[out_edge["to"]]
    vin = (j["x"] - a["x"], j["y"] - a["y"])
    vout = (b["x"] - j["x"], b["y"] - j["y"])
    cross = vin[0] * vout[1] - vin[1] * vout[0]
    dot = vin[0] * vout[0] + vin[1] * vout[1]
    angle = math.degrees(math.atan2(cross, dot))
    if abs(angle) < 35:
        return "s"
    return "l" if angle > 0 else "r"


def generate_connections(
    junctions: dict[str, dict[str, Any]],
    incoming: dict[str, list[dict[str, Any]]],
    outgoing: dict[str, list[dict[str, Any]]],
    via_by_turn: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    """Generate SUMO-style connections for non-U-turn movements."""
    connections: list[dict[str, str]] = []
    for junction_id in junctions:
        for in_edge in incoming.get(junction_id, []):
            for out_edge in outgoing.get(junction_id, []):
                if in_edge["from"] == out_edge["to"]:
                    continue
                max_lane = min(len(in_edge["lanes"]), len(out_edge["lanes"]))
                for lane_index in range(max_lane):
                    conn = {
                        "from": in_edge["id"],
                        "to": out_edge["id"],
                        "fromLane": str(lane_index),
                        "toLane": str(lane_index),
                        "dir": direction_code(in_edge, out_edge, junctions),
                        "state": "M",
                    }
                via = via_by_turn.get((in_edge["id"], out_edge["id"]))
                if via and lane_index == 0:
                    conn["via"] = via
                connections.append(conn)
                if via and lane_index == 0:
                    # SUMO net.xml files also contain the continuation from the
                    # internal edge to the outgoing edge. Without it, sumo can
                    # load routes but fails when starting the simulation.
                    internal_edge_id = via.rsplit("_", 1)[0]
                    connections.append(
                        {
                            "from": internal_edge_id,
                            "to": out_edge["id"],
                            "fromLane": "0",
                            "toLane": str(lane_index),
                            "dir": conn["dir"],
                            "state": conn["state"],
                        }
                    )
    return connections


def junction_shape(x: float, y: float, radius: float = 4.0) -> str:
    """Return a small square-ish junction polygon."""
    return shape_to_text(
        [
            (x - radius, y - radius),
            (x + radius, y - radius),
            (x + radius, y + radius),
            (x - radius, y + radius),
        ]
    )


def build_xml(
    reference: dict[str, Any],
    width_m: float,
    height_m: float,
    priority_junctions: dict[str, dict[str, Any]],
    normal_edges: list[dict[str, Any]],
    internal_edges: list[dict[str, Any]],
    internal_junctions: list[dict[str, Any]],
    connections: list[dict[str, str]],
    incoming: dict[str, list[dict[str, Any]]],
    via_by_turn: dict[tuple[str, str], str],
) -> ET.ElementTree:
    """Build a complete SUMO net.xml tree from generated network objects."""
    root_attrs = {
        "version": reference.get("root_attributes", {}).get("version", "1.0"),
        "junctionCornerDetail": reference.get("root_attributes", {}).get("junctionCornerDetail", "5"),
        "limitTurnSpeed": reference.get("root_attributes", {}).get("limitTurnSpeed", "5.50"),
    }
    schema_key = "{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation"
    if schema_key in reference.get("root_attributes", {}):
        root_attrs[schema_key] = reference["root_attributes"][schema_key]
        ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    root = ET.Element("net", root_attrs)

    all_lane_points: list[tuple[float, float]] = []
    for edge in internal_edges + normal_edges:
        for lane in edge["lanes"]:
            all_lane_points.extend(parse_shape(lane["shape"]))
    if all_lane_points:
        min_lane_x = min(x for x, _ in all_lane_points)
        min_lane_y = min(y for _, y in all_lane_points)
        max_lane_x = max(x for x, _ in all_lane_points)
        max_lane_y = max(y for _, y in all_lane_points)
        conv_min_x = min(0.0, min_lane_x) - 1.0
        conv_min_y = min(0.0, min_lane_y) - 1.0
        conv_max_x = max(width_m, max_lane_x) + 1.0
        conv_max_y = max(height_m, max_lane_y) + 1.0
    else:
        conv_min_x, conv_min_y, conv_max_x, conv_max_y = 0.0, 0.0, width_m, height_m

    ref_location = reference.get("location", {})
    ET.SubElement(
        root,
        "location",
        {
            "netOffset": ref_location.get("netOffset", "0.00,0.00"),
            "convBoundary": f"{fmt(conv_min_x)},{fmt(conv_min_y)},{fmt(conv_max_x)},{fmt(conv_max_y)}",
            "origBoundary": ref_location.get("origBoundary", f"0.000000,0.000000,{fmt(width_m)},{fmt(height_m)}"),
            "projParameter": ref_location.get("projParameter", DEFAULT_PROJ_PARAMETER),
        },
    )

    for type_attrs in reference["types"]:
        ET.SubElement(root, "type", copy.deepcopy(type_attrs))

    for edge in internal_edges + normal_edges:
        edge_attrs = {key: str(value) for key, value in edge.items() if key != "lanes"}
        edge_elem = ET.SubElement(root, "edge", edge_attrs)
        for lane in edge["lanes"]:
            ET.SubElement(edge_elem, "lane", {key: str(value) for key, value in lane.items()})

    normal_edge_to_junction = {edge["id"]: edge["to"] for edge in normal_edges}
    request_count_by_junction: Counter[str] = Counter()
    for connection in connections:
        junction_id = normal_edge_to_junction.get(connection["from"])
        if junction_id:
            request_count_by_junction[junction_id] += 1

    for junction_id, junction in priority_junctions.items():
        inc_lanes = [edge["lanes"][0]["id"] for edge in incoming.get(junction_id, [])]
        int_lanes = [
            lane_id
            for (from_edge, _to_edge), lane_id in via_by_turn.items()
            if any(edge["id"] == from_edge for edge in incoming.get(junction_id, []))
        ]
        junction_elem = ET.SubElement(
            root,
            "junction",
            {
                "id": junction_id,
                "type": "priority",
                "x": fmt(junction["x"]),
                "y": fmt(junction["y"]),
                "incLanes": " ".join(inc_lanes),
                "intLanes": " ".join(int_lanes),
                "shape": junction_shape(junction["x"], junction["y"]),
            },
        )
        request_count = max(1, request_count_by_junction.get(junction_id, 0))
        request_bits = "0" * request_count
        for request_index in range(request_count):
            ET.SubElement(
                junction_elem,
                "request",
                {
                    "index": str(request_index),
                    "response": request_bits,
                    "foes": request_bits,
                    "cont": "0",
                },
            )

    for junction in internal_junctions:
        ET.SubElement(root, "junction", {key: str(value) for key, value in junction.items()})

    for connection in connections:
        ET.SubElement(root, "connection", connection)

    return ET.ElementTree(root)


def validate_network(tree_or_path: ET.ElementTree | str | Path) -> dict[str, int]:
    """Validate generated SUMO net.xml structure and print summary statistics."""
    if isinstance(tree_or_path, ET.ElementTree):
        tree = tree_or_path
    else:
        try:
            tree = ET.parse(tree_or_path)
        except ET.ParseError as exc:
            raise ValueError(f"Generated XML is not parseable: {exc}") from exc

    root = tree.getroot()
    if root.tag != "net":
        raise ValueError(f"Generated XML root must be <net>, got <{root.tag}>")

    location = root.find("location")
    if location is None:
        raise ValueError("Missing required <location> element")
    conv_boundary = location.get("convBoundary")
    if not conv_boundary:
        raise ValueError("<location> is missing convBoundary")
    min_x, min_y, max_x, max_y = parse_boundary(conv_boundary)

    edge_ids: set[str] = set()
    lane_ids: set[str] = set()
    junction_ids: set[str] = set()
    lane_count = 0
    normal_edge_count = 0
    internal_edge_count = 0

    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id:
            raise ValueError("An <edge> is missing id")
        if edge_id in edge_ids:
            raise ValueError(f"Duplicate edge id: {edge_id}")
        edge_ids.add(edge_id)
        if edge.get("function") == "internal":
            internal_edge_count += 1
        else:
            normal_edge_count += 1
            if not edge.get("from") or not edge.get("to"):
                raise ValueError(f"Normal edge {edge_id} must have from/to attributes")

        for lane in edge.findall("lane"):
            lane_count += 1
            lane_id = lane.get("id")
            if not lane_id:
                raise ValueError(f"Lane in edge {edge_id} is missing id")
            if lane_id in lane_ids:
                raise ValueError(f"Duplicate lane id: {lane_id}")
            lane_ids.add(lane_id)
            try:
                length = float(lane.get("length", "0"))
            except ValueError as exc:
                raise ValueError(f"Lane {lane_id} has invalid length: {lane.get('length')}") from exc
            if length <= 0:
                raise ValueError(f"Lane {lane_id} has non-positive length: {length}")
            shape = lane.get("shape")
            if not shape:
                raise ValueError(f"Lane {lane_id} is missing shape")
            for x, y in parse_shape(shape):
                if not (min_x - 0.01 <= x <= max_x + 0.01 and min_y - 0.01 <= y <= max_y + 0.01):
                    raise ValueError(
                        f"Lane {lane_id} coordinate ({x:.2f}, {y:.2f}) is outside convBoundary {conv_boundary}"
                    )

    priority_junction_count = 0
    internal_junction_count = 0
    for junction in root.findall("junction"):
        junction_id = junction.get("id")
        if not junction_id:
            raise ValueError("A <junction> is missing id")
        if junction_id in junction_ids:
            raise ValueError(f"Duplicate junction id: {junction_id}")
        junction_ids.add(junction_id)
        if junction.get("type") == "priority":
            priority_junction_count += 1
        elif junction.get("type") == "internal":
            internal_junction_count += 1

    connection_count = 0
    for connection in root.findall("connection"):
        connection_count += 1
        from_edge = connection.get("from")
        to_edge = connection.get("to")
        if from_edge not in edge_ids:
            raise ValueError(f"Connection references missing from edge: {from_edge}")
        if to_edge not in edge_ids:
            raise ValueError(f"Connection references missing to edge: {to_edge}")
        via = connection.get("via")
        if via and via not in lane_ids:
            raise ValueError(f"Connection references missing via lane: {via}")

    stats = {
        "priority_junction_count": priority_junction_count,
        "internal_junction_count": internal_junction_count,
        "total_junction_count": len(junction_ids),
        "normal_edge_count": normal_edge_count,
        "internal_edge_count": internal_edge_count,
        "lane_count": lane_count,
        "connection_count": connection_count,
    }

    print("Validation summary")
    print(f"  priority junctions : {stats['priority_junction_count']}")
    print(f"  internal junctions : {stats['internal_junction_count']}")
    print(f"  total junctions    : {stats['total_junction_count']}")
    print(f"  normal edges       : {stats['normal_edge_count']}")
    print(f"  internal edges     : {stats['internal_edge_count']}")
    print(f"  lanes              : {stats['lane_count']}")
    print(f"  connections        : {stats['connection_count']}")
    return stats


def write_xml(tree: ET.ElementTree, output_path: str | Path) -> None:
    """Pretty-print and write the XML tree with a UTF-8 declaration."""
    root = tree.getroot()
    ET.indent(root, space="    ")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True) if output.parent != Path(".") else None
    xml_body = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    output.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body, encoding="UTF-8")


def finalize_with_netconvert(raw_path: str | Path, output_path: str | Path) -> bool:
    """Rebuild a generated net.xml with SUMO netconvert so junction logic is executable.

    The Python generator creates a readable SUMO-like network. SUMO's simulator
    also expects internally consistent link indices and junction logic. Calling
    netconvert as the final normalization step makes the output suitable for
    randomTrips.py, duarouter, and sumo.
    """
    raw = Path(raw_path)
    output = Path(output_path)
    command = ["netconvert", "-s", str(raw), "-o", str(output), "--no-warnings"]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return False
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"netconvert failed while rebuilding {output}: {detail}")
    return True


def ensure_sumo_cfg(
    cfg_path: str | Path,
    net_file: str = "map.net.xml",
    route_file: str = "map.rou.xml",
    fcd_file: str = "map.xml",
    end_time: int = 200,
    max_num_vehicles: int = 100,
) -> None:
    """Create a simple SUMO configuration file when one is not already present."""
    cfg = Path(cfg_path)
    if cfg.exists():
        return
    cfg.write_text(
        f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>

<configuration>
  <input>
    <net-file value=\"{net_file}\"/>
    <route-files value=\"{route_file}\"/>
  </input>
  <time>
    <begin value=\"0\"/>
    <end value=\"{end_time}\"/>
  </time>
  <processing>
    <max-num-vehicles value=\"{max_num_vehicles}\"/>
  </processing>
  <output>
    <fcd-output value=\"{fcd_file}\"/>
  </output>
  <gui_only>
    <start value=\"true\"/>
  </gui_only>
</configuration>
""",
        encoding="UTF-8",
    )


def run_command(command: list[str], cwd: Path) -> str:
    """Run a SUMO command and raise a clear error when it fails."""
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, shell=False)
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed in "
            f"{cwd}:\n  {' '.join(command)}\n\n{output.strip()}"
        )
    return output


def run_sumo_pipeline(
    net_path: str | Path,
    period: float = 0.3,
    end_time: int = 200,
    seed: int = 1,
    cfg_name: str = "map.sumo.cfg",
) -> list[tuple[str, str]]:
    """Run randomTrips.py, duarouter, and sumo for a generated net.xml."""
    net = Path(net_path)
    work_dir = net.parent
    net_name = net.name
    trips_name = "map.trips.xml"
    route_name = "map.rou.xml"
    fcd_name = "map.xml"
    cfg_path = work_dir / cfg_name

    ensure_sumo_cfg(cfg_path, net_name, route_name, fcd_name, end_time)
    random_trips = shutil.which("randomTrips.py") or "randomTrips.py"
    if str(random_trips).lower().endswith(".py"):
        # In the PyInstaller GUI build, sys.executable points to
        # SUMO_Net_Generator.exe itself. Use a real Python interpreter so the
        # helper script does not relaunch this GUI.
        python_exe = shutil.which("python") or shutil.which("python3")
        if python_exe is None and not getattr(sys, "frozen", False):
            python_exe = sys.executable
        if python_exe is None:
            raise RuntimeError("Could not find python.exe to run randomTrips.py")
        random_trips_command = [python_exe, random_trips]
    else:
        random_trips_command = [random_trips]

    commands = [
        (
            "randomTrips.py",
            random_trips_command
            + [
                "-n",
                net_name,
                "-o",
                trips_name,
                "-r",
                route_name,
                "-p",
                str(period),
                "-e",
                str(end_time),
                "--seed",
                str(seed),
            ],
        ),
        (
            "duarouter",
            [
                "duarouter",
                "-n",
                net_name,
                "--route-files",
                trips_name,
                "-o",
                route_name,
                "--ignore-errors",
            ],
        ),
        (
            "sumo",
            [
                "sumo",
                "-c",
                cfg_name,
                "--fcd-output",
                fcd_name,
                "--end",
                str(end_time),
            ],
        ),
    ]

    logs: list[tuple[str, str]] = []
    for name, command in commands:
        logs.append((name, run_command(command, work_dir)))
    return logs


def inspect_final_network(
    net_path: str | Path,
    reject_straight_degree_two: bool = True,
    reject_large_blank_areas: bool = True,
) -> dict[str, Any]:
    """Inspect a finalized net.xml and reject visual dead-end/non-intersection dots."""
    root = ET.parse(net_path).getroot()
    normal_edges = [edge for edge in root.findall("edge") if edge.get("function") != "internal"]
    internal_edges = [edge for edge in root.findall("edge") if edge.get("function") == "internal"]
    junctions = root.findall("junction")
    priority = [junction for junction in junctions if junction.get("type") == "priority"]
    internal = [junction for junction in junctions if junction.get("type") == "internal"]
    bad_types = [junction.get("id") for junction in junctions if junction.get("type") in {"dead_end", "unregulated"}]

    positions = {
        junction.get("id"): (float(junction.get("x", "0")), float(junction.get("y", "0")))
        for junction in priority
        if junction.get("id")
    }
    adjacency: dict[str, set[str]] = {junction_id: set() for junction_id in positions}
    for edge in normal_edges:
        from_id = edge.get("from")
        to_id = edge.get("to")
        if from_id in adjacency and to_id in adjacency:
            adjacency[from_id].add(to_id)
            adjacency[to_id].add(from_id)

    dead_like = [junction_id for junction_id, neighbors in adjacency.items() if len(neighbors) <= 1]
    straight_degree_two: list[str] = []
    if reject_straight_degree_two:
        for junction_id, neighbors in adjacency.items():
            if len(neighbors) != 2:
                continue
            n1, n2 = list(neighbors)
            x, y = positions[junction_id]
            x1, y1 = positions[n1]
            x2, y2 = positions[n2]
            v1 = (x1 - x, y1 - y)
            v2 = (x2 - x, y2 - y)
            len1 = math.hypot(*v1)
            len2 = math.hypot(*v2)
            if len1 <= 0 or len2 <= 0:
                straight_degree_two.append(junction_id)
                continue
            cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
            if cos_angle < -0.95:
                straight_degree_two.append(junction_id)

    seen: set[str] = set()
    if adjacency:
        stack = [next(iter(adjacency))]
        seen.add(stack[0])
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

    sparse_tiles: list[tuple[int, int, int]] = []
    if reject_large_blank_areas and positions:
        location = root.find("location")
        if location is not None and location.get("convBoundary"):
            min_x, min_y, max_x, max_y = parse_boundary(location.get("convBoundary", "0,0,1,1"))
        else:
            min_x = min(x for x, _ in positions.values())
            min_y = min(y for _, y in positions.values())
            max_x = max(x for x, _ in positions.values())
            max_y = max(y for _, y in positions.values())
        tile_cols = 4
        tile_rows = 4
        counts = [[0 for _ in range(tile_cols)] for _ in range(tile_rows)]
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)
        for x, y in positions.values():
            col = min(tile_cols - 1, max(0, int((x - min_x) / width * tile_cols)))
            row = min(tile_rows - 1, max(0, int((y - min_y) / height * tile_rows)))
            counts[row][col] += 1
        for row in range(tile_rows):
            for col in range(tile_cols):
                if counts[row][col] < 2:
                    sparse_tiles.append((row, col, counts[row][col]))

    stats = {
        "priority_junction_count": len(priority),
        "internal_junction_count": len(internal),
        "total_junction_count": len(junctions),
        "normal_edge_count": len(normal_edges),
        "internal_edge_count": len(internal_edges),
        "lane_count": sum(len(edge.findall("lane")) for edge in root.findall("edge")),
        "connection_count": len(root.findall("connection")),
        "normal_edge_lane_distribution": Counter(len(edge.findall("lane")) for edge in normal_edges),
        "dead_like_junctions": dead_like,
        "straight_degree_two_junctions": straight_degree_two,
        "sparse_tiles": sparse_tiles,
        "bad_type_junctions": bad_types,
        "connected_priority_graph": len(seen) == len(priority),
    }

    problems = []
    if bad_types:
        problems.append(f"bad junction types: {bad_types[:8]}")
    if dead_like:
        problems.append(f"dead-like priority junctions: {dead_like[:8]}")
    if straight_degree_two:
        problems.append(f"straight degree-2 priority junctions: {straight_degree_two[:8]}")
    if sparse_tiles:
        problems.append(f"large blank/sparse map areas: {sparse_tiles[:8]}")
    if not stats["connected_priority_graph"]:
        problems.append("priority junction graph is disconnected")
    if stats["normal_edge_lane_distribution"] != Counter({1: len(normal_edges)}):
        problems.append(f"normal edge lanes are not one-per-direction: {stats['normal_edge_lane_distribution']}")
    if problems:
        raise ValueError("Final net.xml quality check failed: " + "; ".join(problems))

    return stats


def generate_map_file(
    reference: dict[str, Any],
    output_path: str | Path,
    width_m: float,
    height_m: float,
    rows: int,
    cols: int,
    junction_target: int,
    internal_junction_target: int,
    road_missing_prob: float,
    jitter_m: float,
    seed: int,
    skip_netconvert: bool = False,
    vary_layout: bool = False,
    layout_variation: float = 0.35,
) -> dict[str, Any]:
    """Generate one finalized map file and return final net.xml statistics."""
    effective_missing_prob = min(road_missing_prob, 0.03) if vary_layout else road_missing_prob
    tried_probs = [effective_missing_prob]
    if effective_missing_prob > 0:
        tried_probs.extend([effective_missing_prob * 0.5, 0.0])
    last_error: Exception | None = None

    max_attempts = 10 if vary_layout else 1
    for attempt in range(max_attempts):
        for current_missing_prob in tried_probs:
            try:
                current_rows, current_cols = (
                    choose_varied_layout(rows, cols, junction_target, seed, attempt) if vary_layout else (rows, cols)
                )
                attempt_seed = seed + attempt * 1009 if vary_layout else seed
                attempt_variation_scale = max(0.20, 1.0 - attempt * 0.08)
                rng = random.Random(seed + 707 + attempt * 37)
                current_jitter = jitter_m
                current_layout_variation = layout_variation
                if vary_layout:
                    current_jitter = jitter_m * rng.uniform(0.75, 1.35) * attempt_variation_scale
                    current_layout_variation = min(0.22, layout_variation) * rng.uniform(0.65, 1.0) * attempt_variation_scale
                priority_junctions = generate_junction_grid(
                    width_m,
                    height_m,
                    current_rows,
                    current_cols,
                    junction_target,
                    current_jitter,
                    attempt_seed,
                    current_layout_variation,
                )
                normal_edges, incoming, outgoing = generate_edges(
                    priority_junctions, current_rows, current_cols, current_missing_prob, attempt_seed, reference["type_by_id"]
                )
                turn_candidates = build_turn_candidates(priority_junctions, incoming, outgoing)
                internal_edges, internal_junctions, via_by_turn = generate_internal_edges(
                    priority_junctions, turn_candidates, internal_junction_target, attempt_seed
                )
                connections = generate_connections(priority_junctions, incoming, outgoing, via_by_turn)
                tree = build_xml(
                    reference,
                    width_m,
                    height_m,
                    priority_junctions,
                    normal_edges,
                    internal_edges,
                    internal_junctions,
                    connections,
                    incoming,
                    via_by_turn,
                )

                validate_network(tree)
                output = Path(output_path)
                if skip_netconvert:
                    write_xml(tree, output)
                else:
                    raw_path = output.with_name(output.name + ".raw")
                    write_xml(tree, raw_path)
                    if finalize_with_netconvert(raw_path, output):
                        raw_path.unlink(missing_ok=True)
                    else:
                        raw_path.replace(output)
                        print("WARNING: netconvert was not found. Wrote raw generated XML instead.")
                final_stats = inspect_final_network(output)
                final_stats["used_road_missing_prob"] = current_missing_prob
                final_stats["seed"] = seed
                final_stats["attempt_seed"] = attempt_seed
                final_stats["attempt"] = attempt
                final_stats["rows"] = current_rows
                final_stats["cols"] = current_cols
                final_stats["jitter_m"] = current_jitter
                return final_stats
            except Exception as exc:
                last_error = exc

    raise RuntimeError(f"Could not generate a clean map for seed {seed}: {last_error}")


def choose_varied_layout(base_rows: int, base_cols: int, junction_target: int, seed: int, attempt: int = 0) -> tuple[int, int]:
    """Choose a seed-dependent grid shape while keeping enough cells for the target."""
    rng = random.Random(seed + 606 + attempt * 131)
    candidates: list[tuple[int, int]] = []
    min_rows = max(8, base_rows - 3)
    max_rows = base_rows + 4
    min_cols = max(8, base_cols - 4)
    max_cols = base_cols + 4
    for rows in range(min_rows, max_rows + 1):
        for cols in range(min_cols, max_cols + 1):
            cells = rows * cols
            aspect = cols / max(rows, 1)
            if junction_target <= cells <= junction_target + 12 and 0.85 <= aspect <= 1.35:
                candidates.append((rows, cols))
    if not candidates:
        return base_rows, base_cols

    # Favor noticeably different aspect ratios across seeds without drifting
    # too far from the requested map size.
    candidates.sort(key=lambda item: (abs(item[0] * item[1] - junction_target), abs(item[0] - base_rows)))
    top = candidates[: min(len(candidates), 14)]
    return rng.choice(top)


def print_reference_summary(reference: dict[str, Any], input_path: str | Path) -> None:
    """Print the reference network summary used by CLI and batch mode."""
    print("Reference summary")
    print(f"  input              : {input_path}")
    print(f"  type definitions   : {len(reference['types'])}")
    print(f"  priority junctions : {reference['stats']['priority_junction_count']}")
    print(f"  internal junctions : {reference['stats']['internal_junction_count']}")
    print(f"  normal edges       : {reference['stats']['normal_edge_count']}")
    print(f"  internal edges     : {reference['stats']['internal_edge_count']}")
    print(f"  connections        : {reference['stats']['connection_count']}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic SUMO net.xml that follows a reference net.xml structure."
    )
    parser.add_argument(
        "--input",
        default=r"C:\Users\Choe JongHyeon\Desktop\OSM_project\baseline\map.net.xml",
        help="Reference SUMO net.xml path.",
    )
    parser.add_argument("--output", "--output-path", dest="output_path", default="generated_map.net.xml")
    parser.add_argument("--width-m", type=float, default=412.65)
    parser.add_argument("--height-m", type=float, default=387.65)
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--cols", type=int, default=13)
    parser.add_argument("--junction-target", type=int, default=148)
    parser.add_argument("--internal-junction-target", type=int, default=216)
    parser.add_argument("--road-missing-prob", type=float, default=0.08)
    parser.add_argument("--jitter-m", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vary-layout", action="store_true", help="Vary rows/cols, block spacing, and jitter from the seed.")
    parser.add_argument("--layout-variation", type=float, default=0.35, help="How strongly block sizes vary when layout variation is enabled.")
    parser.add_argument("--run-sumo", action="store_true", help="After generating the net, run randomTrips, duarouter, and sumo.")
    parser.add_argument("--trip-period", type=float, default=0.3)
    parser.add_argument("--trip-end", type=int, default=200)
    parser.add_argument("--trip-seed", type=int, default=1)
    parser.add_argument("--batch-dir", default=None, help="Create many maps under this folder as map_1 ... map_x.")
    parser.add_argument("--batch-count", type=int, default=0, help="Number of maps to generate in batch mode.")
    parser.add_argument("--batch-start-index", type=int, default=1, help="First map index for batch mode.")
    parser.add_argument(
        "--batch-seed-from-index",
        action="store_true",
        default=True,
        help="Use folder index as seed, for example map_5 uses seed 5.",
    )
    parser.add_argument(
        "--skip-netconvert",
        action="store_true",
        help="Write the raw generated XML without the final SUMO netconvert rebuild.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        reference = parse_reference_net(args.input)
        print_reference_summary(reference, args.input)

        if args.batch_count:
            if not args.batch_dir:
                raise ValueError("--batch-dir is required when --batch-count is used")
            if args.batch_count < 1:
                raise ValueError("--batch-count must be at least 1")
            batch_root = Path(args.batch_dir)
            batch_root.mkdir(parents=True, exist_ok=True)
            for index in range(args.batch_start_index, args.batch_start_index + args.batch_count):
                map_dir = batch_root / f"map_{index}"
                map_dir.mkdir(parents=True, exist_ok=True)
                seed = index if args.batch_seed_from_index else args.seed + (index - args.batch_start_index)
                output_path = map_dir / "map.net.xml"
                print(f"\nBatch map_{index}: seed={seed}")
                stats = generate_map_file(
                    reference,
                    output_path,
                    args.width_m,
                    args.height_m,
                    args.rows,
                    args.cols,
                    args.junction_target,
                    args.internal_junction_target,
                    args.road_missing_prob,
                    args.jitter_m,
                    seed,
                    args.skip_netconvert,
                    True,
                    args.layout_variation,
                )
                if args.run_sumo:
                    for name, _log in run_sumo_pipeline(output_path, args.trip_period, args.trip_end, seed):
                        print(f"  {name}: Success")
                print(
                    f"  output={output_path} priority={stats['priority_junction_count']} "
                    f"internal={stats['internal_junction_count']} normal_edges={stats['normal_edge_count']} "
                    f"lanes={dict(stats['normal_edge_lane_distribution'])} "
                    f"layout={stats['rows']}x{stats['cols']}"
                )
            print(f"\nBatch complete: {args.batch_count} maps under {batch_root}")
            return 0

        stats = generate_map_file(
            reference,
            args.output_path,
            args.width_m,
            args.height_m,
            args.rows,
            args.cols,
            args.junction_target,
            args.internal_junction_target,
            args.road_missing_prob,
            args.jitter_m,
            args.seed,
            args.skip_netconvert,
            args.vary_layout,
            args.layout_variation,
        )
        print(f"Generated SUMO net.xml: {args.output_path}")
        if args.run_sumo:
            print("Running SUMO pipeline")
            for name, _log in run_sumo_pipeline(args.output_path, args.trip_period, args.trip_end, args.trip_seed):
                print(f"  {name}: Success")
        print(
            "Done: "
            f"{stats['priority_junction_count']} priority junctions, "
            f"{stats['internal_junction_count']} internal junctions, "
            f"{stats['normal_edge_count']} normal edges, "
            f"lane_distribution={dict(stats['normal_edge_lane_distribution'])}."
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
