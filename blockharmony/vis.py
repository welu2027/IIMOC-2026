#!/usr/bin/env python3
"""
vis.py — Visualization for IIMOC-P2 (Chromatic Block Harmony).

Usage:
    python3 vis.py input.txt output.txt --out out.png

Plots:
  - Array elements as a bar chart (gray bars)
  - Colored block spans as shaded horizontal bands beneath the bars
  - OBJ score, k, and colors used in the title
"""

import sys
import argparse


def parse_input(path):
    with open(path) as f:
        data = f.read().split()
    idx = 0
    n     = int(data[idx]); idx += 1
    B     = int(data[idx]); idx += 1
    alpha = int(data[idx]); idx += 1
    beta  = int(data[idx]); idx += 1
    a     = [int(data[idx + i]) for i in range(n)]
    return n, B, alpha, beta, a


def parse_output(path):
    """Returns list of (l, r, c) 1-indexed, or empty list."""
    blocks = []
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return blocks
    try:
        k = int(lines[0])
    except ValueError:
        return blocks
    for line in lines[1:k+1]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                blocks.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                pass
    return blocks


def compute_obj(blocks, alpha, beta):
    if not blocks:
        return 0, 0, set()
    k = len(blocks)
    colors_used = {c for _, _, c in blocks}
    C = len(colors_used)
    obj = alpha * k - beta * C
    return obj, k, colors_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input",  help="Input file")
    ap.add_argument("output", help="Output file (contestant blocks)")
    ap.add_argument("--out",  default="vis_out.png", help="Output PNG path")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n, B, alpha, beta, a = parse_input(args.input)
    blocks = parse_output(args.output)

    obj, k, colors_used = compute_obj(blocks, alpha, beta)

    # Assign a distinct matplotlib color to each used color index
    cmap = plt.get_cmap("tab10")
    color_rgb = {}
    for i, c in enumerate(sorted(colors_used)):
        color_rgb[c] = cmap(i % 10)

    # Build a per-position color array (None = unblocked)
    pos_color  = [None] * (n + 1)   # 1-indexed
    pos_block  = [None] * (n + 1)   # which block index
    for bi, (l, r, c) in enumerate(blocks):
        for pos in range(l, r + 1):
            pos_color[pos] = c
            pos_block[pos] = bi

    plt.style.use("dark_background")
    fig, (ax_bar, ax_span) = plt.subplots(
        2, 1, figsize=(max(10, n * 0.25), 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.05)

    xs = list(range(1, n + 1))

    # ── Bar chart (top) ──────────────────────────────────────────────────────
    bar_colors = []
    for i in range(1, n + 1):
        c = pos_color[i]
        if c is not None:
            bar_colors.append(color_rgb[c])
        else:
            bar_colors.append("#555555")

    ax_bar.bar(xs, a, color=bar_colors, width=0.85, zorder=3)
    ax_bar.axhline(0, color="white", linewidth=0.5, alpha=0.4)
    # Vertical boundary markers at each block edge
    for l, r, _c in blocks:
        ax_bar.axvline(l - 0.5, color="white", linewidth=0.8, alpha=0.4, zorder=4)
        ax_bar.axvline(r + 0.5, color="white", linewidth=0.8, alpha=0.4, zorder=4)
    ax_bar.set_ylabel("a[i]", fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.15)
    ax_bar.tick_params(axis="x", labelbottom=False)

    # ── Block spans (bottom strip) ───────────────────────────────────────────
    ax_span.set_ylim(0, 1)
    ax_span.set_yticks([])
    ax_span.set_xlabel("index i", fontsize=9)

    for l, r, c in blocks:
        ax_span.axvspan(l - 0.5, r + 0.5, ymin=0.05, ymax=0.95,
                        color=color_rgb[c], alpha=0.7)
        # Tick marks at every block boundary so adjacent same-color blocks
        # are still visually distinct.
        ax_span.axvline(l - 0.5, color="white", linewidth=0.8, alpha=0.6)
        ax_span.axvline(r + 0.5, color="white", linewidth=0.8, alpha=0.6)

    # ── Title ────────────────────────────────────────────────────────────────
    obj_str = f"OBJ = {obj}  (k={k}, colors={len(colors_used)}, α={alpha}, β={beta})"
    fig.suptitle(
        f"IIMOC-P2 | n={n}  B={B}  alpha={alpha}  beta={beta}\n{obj_str}",
        fontsize=10
    )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_patches = []
    for c in sorted(colors_used):
        legend_patches.append(
            mpatches.Patch(color=color_rgb[c], label=f"Color {c}")
        )
    if legend_patches:
        ax_bar.legend(handles=legend_patches, loc="upper right",
                      fontsize=8, framealpha=0.4)

    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[vis] saved -> {args.out}")


if __name__ == "__main__":
    main()
