import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import platform

if platform.system() == "Windows":
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

CSV_PATH = r"D:/Github/Python/연구실/연구 데이터 분석/AppSci (AODV.2025.11.05.v.6)/VANET_Setting (2025.11.05.v.1)/data/fcd_positions_100.csv"
BACKGROUND_PATH = r"D:/Github/Python/연구실/연구 데이터 분석/AppSci (AODV.2025.11.05.v.1)/data/Background.png"
TARGET_IDS = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
TIME_MAX = None
BG_ROI = None
RADIUS = 60.0

ids = [i.strip() for i in TARGET_IDS.split(",") if i.strip()]

df = pd.read_csv(CSV_PATH)
if TIME_MAX is not None:
    df = df[df["time"] <= TIME_MAX]
df = df.sort_values(["time","node"])
df["node"] = df["node"].astype(str)

all_x = df["x"].to_numpy(float)
all_y = df["y"].to_numpy(float)
XMIN, XMAX = np.min(all_x), np.max(all_x)
YMIN, YMAX = np.min(all_y), np.max(all_y)

bg = mpimg.imread(BACKGROUND_PATH)
H, W = bg.shape[:2]
if BG_ROI is None:
    x0, y0, x1, y1 = 0, 0, W, H
else:
    x0, y0, x1, y1 = BG_ROI

def norm_xy(x, y):
    xn = x0 + (x - XMIN) / (XMAX - XMIN) * (x1 - x0)
    yn = y1 - (y - YMIN) / (YMAX - YMIN) * (y1 - y0)
    return xn, yn

def neighbor_counts(df, ids, radius):
    out = {}
    r2 = radius * radius
    for t, g in df.groupby("time"):
        gx = g["x"].to_numpy(float)
        gy = g["y"].to_numpy(float)
        gn = g["node"].to_numpy(str)
        for i, nid in enumerate(gn):
            if nid not in ids:
                continue
            dx = gx - gx[i]
            dy = gy - gy[i]
            d2 = dx*dx + dy*dy
            out[(nid, float(t))] = int(((d2 <= r2) & (d2 > 0)).sum())
    return out

nc_map = neighbor_counts(df, ids, RADIUS)

fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(bg, extent=[0, W, 0, H], origin="lower", zorder=0)

if len(ids) <= 10:
    colors = plt.cm.tab10(np.linspace(0, 1, len(ids)))
elif len(ids) <= 20:
    colors = plt.cm.tab20(np.linspace(0, 1, len(ids)))
else:
    cmap = plt.cm.get_cmap("gist_ncar", len(ids))
    colors = [cmap(i) for i in range(len(ids))]

lines = []
meta = []

for tid, color in zip(ids, colors):
    sub = df[df["node"] == tid].sort_values("time")
    if sub.empty:
        continue
    xs = sub["x"].to_numpy(float)
    ys = sub["y"].to_numpy(float)
    ts = sub["time"].to_numpy(float)
    sp = (sub["speed_kmh"].to_numpy(float) if "speed_kmh" in sub.columns else np.full(len(sub), np.nan))
    xn, yn = norm_xy(xs, ys)
    dummy, = ax.plot(xn, yn, linewidth=1.0, color=color, alpha=0.0, label=f"node {tid}", zorder=1)
    for i in range(len(xn) - 1):
        cnt = nc_map.get((tid, float(ts[i])), 0)
        lw = np.clip(1 + 0.2 * cnt, 1, 6)
        ax.plot(xn[i:i+2], yn[i:i+2], linewidth=lw, color=color, zorder=2)
    ax.scatter(xn[0], yn[0], s=40, color=color, zorder=3)
    ax.scatter(xn[-1], yn[-1], s=40, color=color, marker="x", zorder=3)
    lines.append(dummy)
    meta.append({"node": tid, "time": ts, "speed": sp, "xn": xn, "yn": yn})

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
plt.tight_layout()

ann = ax.annotate("", xy=(0,0), xytext=(12,12), textcoords="offset points",
                  bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.8))
ann.set_visible(False)

def on_move(event):
    if not event.inaxes:
        ann.set_visible(False)
        fig.canvas.draw_idle()
        return
    thr = 8.0
    best = None
    for i, ln in enumerate(lines):
        xdisp, ydisp = ax.transData.transform(np.column_stack([meta[i]["xn"], meta[i]["yn"]])).T
        d = np.hypot(xdisp - event.x, ydisp - event.y)
        j = int(np.argmin(d))
        if d[j] <= thr and (best is None or d[j] < best[0]):
            best = (d[j], i, j)
    if best is None:
        ann.set_visible(False)
        fig.canvas.draw_idle()
        return
    _, li, lj = best
    n = meta[li]["node"]
    t = float(meta[li]["time"][lj])
    s = float(meta[li]["speed"][lj]) if not np.isnan(meta[li]["speed"][lj]) else float("nan")
    cnt = nc_map.get((n, t), np.nan)
    ann.xy = (meta[li]["xn"][lj], meta[li]["yn"][lj])
    ann.set_text(f"node {n}\ntime={t:.0f}s\nNt(<=60m)={cnt}\nspeed={s:.2f} km/h")
    ann.set_visible(True)
    fig.canvas.draw_idle()

cid = fig.canvas.mpl_connect("motion_notify_event", on_move)
plt.show()
