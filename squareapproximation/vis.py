import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ... Keep parse_input, parse_output, union_cells, compute_score as they were ...

def parse_input(path):
    with open(path) as f:
        tokens = f.read().split()
    if not tokens: return 0, 0, []
    idx = 0
    N = int(tokens[idx]); idx += 1
    K = int(tokens[idx]); idx += 1
    rects = []
    for _ in range(N):
        x1, y1, x2, y2 = (int(tokens[idx + k]) for k in range(4))
        idx += 4
        rects.append((x1, y1, x2, y2))
    return N, K, rects

def parse_output(path, K):
    with open(path) as f:
        tokens = f.read().split()
    rects = []
    idx = 0
    for _ in range(K):
        if idx + 3 >= len(tokens):
            break
        x1, y1, x2, y2 = (int(tokens[idx + k]) for k in range(4))
        idx += 4
        rects.append((x1, y1, x2, y2))
    return rects

def union_cells(rects):
    covered = set()
    for (x1, y1, x2, y2) in rects:
        for x in range(x1, x2):
            for y in range(y1, y2):
                covered.add((x, y))
    return covered

def compute_score(in_rects, out_rects):
    setA = union_cells(in_rects)
    setB = union_cells(out_rects)
    areaA = len(setA)
    if areaA == 0:
        return 1.0
    sym_diff = len(setA.symmetric_difference(setB))
    score = 1.0 - sym_diff / areaA
    return max(0.0, score)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input",  help="Input file")
    ap.add_argument("output", help="Output file (contestant rectangles)")
    ap.add_argument("--out",  default="vis_out.png", help="Output PNG path")
    args = ap.parse_args()

    N, K, in_rects  = parse_input(args.input)
    out_rects       = parse_output(args.output, K)

    score = compute_score(in_rects, out_rects)

    all_coords = list(in_rects) + list(out_rects)
    if not all_coords:
        all_coords = [(-1, -1, 1, 1)]
        
    xmin = min(r[0] for r in all_coords) - 2
    xmax = max(r[2] for r in all_coords) + 2
    ymin = min(r[1] for r in all_coords) - 2
    ymax = max(r[3] for r in all_coords) + 2

    setA = union_cells(in_rects)
    setB = union_cells(out_rects)
    both_AB = setA & setB
    only_A  = setA - setB
    only_B  = setB - setA

    plt.style.use("dark_background")
    fig_w = max(8, (xmax - xmin) * 0.02) # Scaled down for reasonable file sizes
    fig_h = max(6, (ymax - ymin) * 0.02)
    fig, ax = plt.subplots(figsize=(min(fig_w, 20), min(fig_h, 14)))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("y", fontsize=9)
    
    # --- FAST PIXEL RENDERING VIA NUMPY ---
    width = xmax - xmin
    height = ymax - ymin
    
    # Create an RGBA image array (transparent by default)
    img = np.zeros((height, width, 4), dtype=np.float32)
    
    # Map sets to coordinates relative to the image bounding box
    def paint_cells(cells, rgba):
        if not cells: return
        # Convert set of tuples to numpy arrays for fast indexing
        cells_arr = np.array(list(cells))
        xs = cells_arr[:, 0] - xmin
        ys = cells_arr[:, 1] - ymin
        img[ys, xs] = rgba

    # Define RGBA colors (values between 0 and 1)
    color_correct = (0.27, 0.86, 0.53, 0.55) # #44dd88
    color_missed  = (0.27, 0.53, 1.0, 0.55)  # #4488ff
    color_extra   = (1.0, 0.53, 0.27, 0.55)  # #ff8844

    paint_cells(both_AB, color_correct)
    paint_cells(only_A, color_missed)
    paint_cells(only_B, color_extra)

    # Use origin="lower" because y increases going up in coordinate geometry
    ax.imshow(img, extent=(xmin, xmax, ymin, ymax), origin="lower", zorder=2)
    # ---------------------------------------

    # Draw outlines (Only drawing K + N outlines is very fast for Matplotlib)
    for (x1, y1, x2, y2) in in_rects:
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0",
            linewidth=1.2, edgecolor="#6699ff", facecolor="none", zorder=3
        )
        ax.add_patch(rect)

    for (x1, y1, x2, y2) in out_rects:
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="square,pad=0",
            linewidth=1.4, edgecolor="#ffaa44", facecolor="none",
            linestyle="--", zorder=4
        )
        ax.add_patch(rect)

    legend_handles = [
        mpatches.Patch(color="#44dd88", alpha=0.7, label="A ∩ B (correct)"),
        mpatches.Patch(color="#4488ff", alpha=0.7, label="A only (missed)"),
        mpatches.Patch(color="#ff8844", alpha=0.7, label="B only (extra)"),
        mpatches.Patch(edgecolor="#6699ffff", facecolor="none", linewidth=1.2, label="Input rects (A)"),
        mpatches.Patch(edgecolor="#ffaa44", facecolor="none", linewidth=1.4, linestyle="--", label="Output rects (B)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.4)

    fig.suptitle(
        f"IIMOC-P4 | N={N}  K={K}  score={score:.4f}\n"
        f"|A|={len(setA)}  |B|={len(setB)}  |A∩B|={len(both_AB)}  "
        f"|A△B|={len(only_A)+len(only_B)}",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[vis] saved -> {args.out}")

if __name__ == "__main__":
    main()