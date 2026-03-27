# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:55:44 2026

@author: Choe JongHyeon
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

base_dir = Path(r'C:\Users\Choe JongHyeon\Desktop\map\RERR_SINR\backup_8')
node_list = [3, 9, 16, 21, 24, 29, 30, 33, 35, 51, 50, 59, 60, 61, 66, 69, 75, 76, 82, 83, 89, 90, 92, 94, 101, 104, 106]
maps = [1, 3]
packet_types = ["aodv::Rreq", "aodv::Rrep", "aodv::Rerr"]

def read_sinr_metrics(csv_path: Path, packet_type: str):
    with csv_path.open('r', encoding='utf-8', errors='ignore', newline='') as f:
        first = f.readline()
        if not first:
            return None

        delim = "\t" if "\t" in first else ","
        headers = [h.strip() for h in first.strip().split(delim)]
        idx = {h: i for i, h in enumerate(headers)}

        required = ["SINR", "Success", "Packet"]
        if any(col not in idx for col in required):
            return None

        total = 0
        fail = 0
        sinrs = []

        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            if not row or len(row) <= max(idx["SINR"], idx["Success"], idx["Packet"]):
                continue

            pkt = str(row[idx["Packet"]]).strip()
            if pkt != packet_type:
                continue

            s_sinr = str(row[idx["SINR"]]).strip()
            if s_sinr == "":
                continue

            try:
                v = float(s_sinr)
            except:
                continue

            s_succ = str(row[idx["Success"]]).strip().upper()
            total += 1
            if s_succ == "FALSE":
                fail += 1
            sinrs.append(v)

        if total == 0:
            return None

        sinrs.sort()
        n = len(sinrs)
        if n % 2 == 1:
            med = sinrs[n // 2]
        else:
            med = (sinrs[n // 2 - 1] + sinrs[n // 2]) / 2

        return fail / total, med

def collect():
    result = {
        pkt: {
            "fail_by_map": {m: [] for m in maps},
            "medsinr_by_map": {m: [] for m in maps},
        }
        for pkt in packet_types
    }

    for m in maps:
        for exp in node_list:
            p = base_dir / str(m) / "map_4" / str(exp) / "sinr_all_log.csv"
            if not p.is_file():
                continue

            for pkt in packet_types:
                out = read_sinr_metrics(p, pkt)
                if out is None:
                    continue
                fr, med = out
                result[pkt]["fail_by_map"][m].append(fr)
                result[pkt]["medsinr_by_map"][m].append(med)

    return result

def safe_boxplot(ax, data, positions, width=0.6):
    valid_data = []
    valid_positions = []
    for d, p in zip(data, positions):
        if d:
            valid_data.append(d)
            valid_positions.append(p)
    if valid_data:
        ax.boxplot(valid_data, positions=valid_positions, widths=width, showfliers=True)

def plot_combined(result):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    label_map = {
        "aodv::Rreq": "RREQ",
        "aodv::Rrep": "RREP",
        "aodv::Rerr": "RERR"
    }

    for col, pkt in enumerate(packet_types):
        fail_data = [result[pkt]["fail_by_map"][m] for m in maps]
        med_data = [result[pkt]["medsinr_by_map"][m] for m in maps]

        ax1 = axes[0, col]
        safe_boxplot(ax1, fail_data, positions=range(1, len(maps) + 1))
        ax1.set_xticks(range(1, len(maps) + 1))
        ax1.set_xticklabels(maps)
        ax1.set_xlabel("Map")
        ax1.set_ylabel("Fail rate")
        ax1.set_title(f"{label_map[pkt]} Fail Rate")
        ax1.grid(True)

        ax2 = axes[1, col]
        safe_boxplot(ax2, med_data, positions=range(1, len(maps) + 1))
        ax2.set_xticks(range(1, len(maps) + 1))
        ax2.set_xticklabels(maps)
        ax2.set_xlabel("Map")
        ax2.set_ylabel("Median SINR")
        ax2.set_title(f"{label_map[pkt]} Median SINR")
        ax2.grid(True)

    plt.tight_layout()
    plt.show()

result = collect()

for pkt in packet_types:
    for m in maps:
        print(pkt, "map", m, "n_failrate =", len(result[pkt]["fail_by_map"][m]), "n_medsinr =", len(result[pkt]["medsinr_by_map"][m]))

plot_combined(result)