#!/usr/bin/env python3
"""
Rectangle Approximation — greedy build-up with full coordinate enumeration.

Strategy:
- For small N (total coord combos <= threshold): global build-up over all rects.
  Enumerates ALL (x1,x2,y1,y2) from input coordinates using 2D prefix sums.
  This finds cross-cluster merges (key for quality on small inputs).
- For larger N: per-cluster build-up with restricted candidate sets.
  Uses input rects + nearby-pair bboxes; input rects only for huge clusters.
- Budget allocated via greedy marginal gain (greedy knapsack).
- Post-process: shrink each output rect inward when it helps.
- Global time budget prevents TLE.
"""
import sys
import random
import time
import heapq

_T0 = time.monotonic()
TIME_LIMIT = 4.2


def read_input():
    data = sys.stdin.buffer.read().split()
    p = 0
    N = int(data[p]); p += 1
    K = int(data[p]); p += 1
    rects = []
    for _ in range(N):
        rects.append((int(data[p]), int(data[p+1]),
                      int(data[p+2]), int(data[p+3])))
        p += 4
    return N, K, rects


def overlap_interior(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def find_clusters(rects):
    n = len(rects)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        ai = rects[i]
        for j in range(i + 1, n):
            if overlap_interior(ai, rects[j]):
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def max_sum_subrect(val, nx, ny):
    """2D Kadane: maximum-sum subrectangle of grid val (nx x ny).

    For every pair of x-rows (i1..i2) we accumulate column sums and run 1D
    Kadane over y. O(nx^2 * ny). Returns (best_sum, (i1, i2, j1, j2)) with
    i2/j2 exclusive, or (0, None) if no positive-sum rectangle exists.
    """
    best = 0
    res = None
    for i1 in range(nx):
        col = [0] * ny
        for i2 in range(i1, nx):
            rv = val[i2]
            cur = 0
            start = 0
            for j in range(ny):
                col[j] += rv[j]
                c = col[j]
                if cur <= 0:
                    cur = c
                    start = j
                else:
                    cur += c
                if cur > best:
                    best = cur
                    res = (i1, i2 + 1, start, j + 1)
    return best, res


def largest_rect_in_A(avail, nx, ny, cell_dx, cell_dy):
    """Largest-area axis-aligned rectangle whose cells are ALL available (in A
    and not yet covered). Zero-excess by construction.

    Sweeps x-rows maintaining, per y-column, the accumulated x-width of the
    consecutive available run; each row solves a variable-width largest-
    rectangle-in-histogram (bar height = run width in x, bar width = cell_dy).
    O(nx * ny). Returns (best_area, (i1, i2, j1, j2)) or (0, None).
    """
    best = 0
    res = None
    w = [0.0] * ny
    startx = [0] * ny
    ywidth_at = [0.0] * (ny + 1)
    for j in range(ny):
        ywidth_at[j + 1] = ywidth_at[j] + cell_dy[j]
    for i in range(nx):
        ai = avail[i]
        dx = cell_dx[i]
        for j in range(ny):
            if ai[j]:
                if w[j] == 0.0:
                    startx[j] = i
                w[j] += dx
            else:
                w[j] = 0.0
        st = []  # (height_w, ystart_index, xstart_row)
        for j in range(ny):
            h = w[j]
            if not st or h > st[-1][0]:
                st.append((h, j, startx[j]))
            elif h < st[-1][0]:
                jstart = j
                xs_row = startx[j]
                while st and st[-1][0] > h:
                    ph, pj, pxr = st.pop()
                    area = ph * (ywidth_at[j] - ywidth_at[pj])
                    if area > best:
                        best = area
                        res = (pxr, i + 1, pj, j)
                    jstart = pj
                    xs_row = pxr
                st.append((h, jstart, xs_row))
        for ph, pj, pxr in reversed(st):
            area = ph * (ywidth_at[ny] - ywidth_at[pj])
            if area > best:
                best = area
                res = (pxr, i + 1, pj, ny)
    return best, res


def refine_cluster(cluster_rects, out_rects, deadline):
    """Local search: repeatedly remove the output rect with the least exclusive
    in-A coverage and re-place the largest rectangle that now fits in the
    uncovered region. A swap is kept only when the new rect covers strictly
    more new in-A area than the removed rect exclusively held — i.e. freeing a
    redundant rect let a bigger rectangle form elsewhere. Zero excess preserved.
    """
    b = len(out_rects)
    if b == 0:
        return out_rects

    xs_set = set(); ys_set = set()
    for r in list(cluster_rects) + list(out_rects):
        xs_set.add(r[0]); xs_set.add(r[2])
        ys_set.add(r[1]); ys_set.add(r[3])
    xs = sorted(xs_set); ys = sorted(ys_set)
    nx = len(xs) - 1; ny = len(ys) - 1
    if nx <= 0 or ny <= 0:
        return out_rects
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    if nx * ny > CELL_CAP or _grid_work(cluster_rects, xi, yi) > WORK_CAP:
        return out_rects
    cdx = [xs[i + 1] - xs[i] for i in range(nx)]
    cdy = [ys[j + 1] - ys[j] for j in range(ny)]

    inA = [[False] * ny for _ in range(nx)]
    for r in cluster_rects:
        for i in range(xi[r[0]], xi[r[2]]):
            ra = inA[i]
            for j in range(yi[r[1]], yi[r[3]]):
                ra[j] = True

    coverage = [[0] * ny for _ in range(nx)]
    bounds = []
    for r in out_rects:
        bd = (xi[r[0]], xi[r[2]], yi[r[1]], yi[r[3]])
        bounds.append(bd)
        for i in range(bd[0], bd[1]):
            cr = coverage[i]
            for j in range(bd[2], bd[3]):
                cr[j] += 1
    avail = [[inA[i][j] and coverage[i][j] == 0 for j in range(ny)]
             for i in range(nx)]
    cur = list(out_rects)

    def excl(bd):
        i1, i2, j1, j2 = bd; s = 0
        for i in range(i1, i2):
            cr = coverage[i]; ra = inA[i]; dx = cdx[i]
            for j in range(j1, j2):
                if ra[j] and cr[j] == 1:
                    s += dx * cdy[j]
        return s

    excls = [excl(bd) for bd in bounds]

    improved = True
    while improved and time.monotonic() < deadline:
        improved = False
        order = sorted(range(len(cur)), key=lambda i: excls[i])
        for idx in order:
            if time.monotonic() >= deadline:
                break
            i1, i2, j1, j2 = bounds[idx]
            E = excls[idx]
            for i in range(i1, i2):
                cr = coverage[i]; ra = inA[i]; av = avail[i]
                for j in range(j1, j2):
                    cr[j] -= 1
                    if cr[j] == 0 and ra[j]:
                        av[j] = True
            G, res = largest_rect_in_A(avail, nx, ny, cdx, cdy)
            if res is not None and G > E + 1e-9:
                ni1, ni2, nj1, nj2 = res
                for i in range(ni1, ni2):
                    cr = coverage[i]; av = avail[i]
                    for j in range(nj1, nj2):
                        cr[j] += 1; av[j] = False
                bounds[idx] = res
                cur[idx] = (xs[ni1], ys[nj1], xs[ni2], ys[nj2])
                excls = [excl(bd) for bd in bounds]
                improved = True
            else:
                for i in range(i1, i2):
                    cr = coverage[i]; av = avail[i]
                    for j in range(j1, j2):
                        cr[j] += 1; av[j] = False
    return cur


# build_up's per-step primitive. Zero-excess maximal rectangles
# (largest_rect_in_A) empirically beat max-gain Kadane on the judge
# distribution: greedy beneficial-excess does not pay off globally, so we
# default to histogram everywhere. Kadane is still used when all_combos is
# forced (tiny global-feasible inputs).
KADANE_THRESH = 0


# Defensive caps. The per-cell passes (grid construction, histogram, edge
# sliding) cost O(nx*ny) and O(sum of compressed rect areas). Clusters this
# large do not arise in the judge distribution (gen.py shrinks rectangles as N
# grows), but a pathological dense cluster could blow the time limit, so any
# cluster above these caps falls back to a cheap rect selection.
CELL_CAP = 250000
WORK_CAP = 4000000


def _grid_work(rects, xi, yi):
    """Total compressed-grid cell writes needed to rasterise rects (O(len))."""
    w = 0
    for r in rects:
        w += (xi[r[2]] - xi[r[0]]) * (yi[r[3]] - yi[r[1]])
        if w > WORK_CAP:
            break
    return w


def _largest_rects_fallback(input_rects, k):
    """Fast, grid-free fallback: the k largest-area input rectangles."""
    ordered = sorted(input_rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]),
                     reverse=True)
    out = ordered[:k]
    gains = [(r[2] - r[0]) * (r[3] - r[1]) for r in out]
    while len(out) < k:
        out.append(input_rects[0]); gains.append(0)
    return list(out), gains


def build_up(input_rects, k, all_combos=False):
    """
    Greedy build-up: at each step add the rectangle that most reduces |A△B|.

    gain(R) = area(R ∩ A-not-yet-covered) − area(R ∩ not-A-not-yet-covered).

    The best such rectangle is the maximum-sum subrectangle of a signed grid
    (+cell area inside A, −cell area outside A). For grids where that is
    affordable we use 2D Kadane (max_sum_subrect), which lets a rect take on
    beneficial excess. For very large grids we fall back to the largest fully-
    inside-A rectangle (largest_rect_in_A): zero excess, O(nx*ny) per step.

    Returns (output_rects, marginal_gains), both length k.
    """
    n = len(input_rects)
    if k == 0:
        return [], []

    # --- Coordinate compression ---
    xs_set = set(); ys_set = set()
    for r in input_rects:
        xs_set.add(r[0]); xs_set.add(r[2])
        ys_set.add(r[1]); ys_set.add(r[3])
    xs = sorted(xs_set); ys = sorted(ys_set)
    nx = len(xs) - 1; ny = len(ys) - 1
    if nx <= 0 or ny <= 0:
        out = [input_rects[0]] * k
        return out, [0] * k

    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}

    if nx * ny > CELL_CAP or _grid_work(input_rects, xi, yi) > WORK_CAP:
        return _largest_rects_fallback(input_rects, k)

    cell_dx = [xs[i + 1] - xs[i] for i in range(nx)]
    cell_dy = [ys[j + 1] - ys[j] for j in range(ny)]

    # --- in-A grid ---
    inA = [[False] * ny for _ in range(nx)]
    for r in input_rects:
        i1, i2, j1, j2 = xi[r[0]], xi[r[2]], yi[r[1]], yi[r[3]]
        for i in range(i1, i2):
            row = inA[i]
            for j in range(j1, j2):
                row[j] = True

    output_rects = []
    gains = []
    use_kadane = all_combos or (nx * nx * ny <= KADANE_THRESH)

    if use_kadane:
        # Signed value grid; covered cells set to 0 (no further gain or loss).
        val = [[0] * ny for _ in range(nx)]
        for i in range(nx):
            dx = cell_dx[i]; ra = inA[i]; rv = val[i]
            for j in range(ny):
                a = dx * cell_dy[j]
                rv[j] = a if ra[j] else -a
        for _ in range(k):
            if time.monotonic() - _T0 > TIME_LIMIT:
                break
            best, res = max_sum_subrect(val, nx, ny)
            if res is None or best <= 0:
                break
            i1, i2, j1, j2 = res
            gains.append(best)
            output_rects.append((xs[i1], ys[j1], xs[i2], ys[j2]))
            for i in range(i1, i2):
                rv = val[i]
                for j in range(j1, j2):
                    rv[j] = 0
    else:
        avail = [row[:] for row in inA]  # in A and not yet covered
        for _ in range(k):
            if time.monotonic() - _T0 > TIME_LIMIT:
                break
            best, res = largest_rect_in_A(avail, nx, ny, cell_dx, cell_dy)
            if res is None or best <= 0:
                break
            i1, i2, j1, j2 = res
            gains.append(best)
            output_rects.append((xs[i1], ys[j1], xs[i2], ys[j2]))
            for i in range(i1, i2):
                ra = avail[i]
                for j in range(j1, j2):
                    ra[j] = False

    while len(output_rects) < k:
        output_rects.append(output_rects[0] if output_rects else input_rects[0])
        gains.append(0)

    return output_rects, gains


def optimize_edges(output_rects, ref_rects):
    """Slide each output rect's four edges to their best positions.

    An edge may move outward (grow, swallowing adjacent uncovered in-A cells at
    the cost of any excess crossed) or inward (shrink, trimming excess). For one
    edge the others are held fixed and the position minimising |A△B| is chosen;
    only strictly improving moves are applied, so this can never lower the score.

    Per-cell rules (signed = +area if in A else −area; coverage = #rects over it):
      adding an uncovered cell (coverage 0) changes symdiff by −signed;
      removing an exclusive cell (coverage 1) changes symdiff by +signed;
      cells covered by another rect contribute 0 either way.
    """
    if not output_rects or not ref_rects:
        return list(output_rects)

    xs_set = set(); ys_set = set()
    for r in list(ref_rects) + list(output_rects):
        xs_set.add(r[0]); xs_set.add(r[2])
        ys_set.add(r[1]); ys_set.add(r[3])
    xs = sorted(xs_set); ys = sorted(ys_set)
    nx = len(xs) - 1; ny = len(ys) - 1
    if nx <= 0 or ny <= 0:
        return list(output_rects)

    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    if nx * ny > CELL_CAP or _grid_work(ref_rects, xi, yi) > WORK_CAP:
        return list(output_rects)
    cell_dx = [xs[i + 1] - xs[i] for i in range(nx)]
    cell_dy = [ys[j + 1] - ys[j] for j in range(ny)]

    A_grid = [[False] * ny for _ in range(nx)]
    for r in ref_rects:
        i1, i2, j1, j2 = xi[r[0]], xi[r[2]], yi[r[1]], yi[r[3]]
        for i in range(i1, i2):
            row = A_grid[i]
            for j in range(j1, j2):
                row[j] = True

    signed = [[0] * ny for _ in range(nx)]
    for i in range(nx):
        dx = cell_dx[i]
        row_a = A_grid[i]; row_s = signed[i]
        for j in range(ny):
            a = dx * cell_dy[j]
            row_s[j] = a if row_a[j] else -a

    coverage = [[0] * ny for _ in range(nx)]
    out_bounds = []
    current = list(output_rects)
    for r in current:
        i1 = xi.get(r[0]); i2 = xi.get(r[2])
        j1 = yi.get(r[1]); j2 = yi.get(r[3])
        if i1 is None or i2 is None or j1 is None or j2 is None:
            out_bounds.append(None); continue
        out_bounds.append((i1, i2, j1, j2))
        for i in range(i1, i2):
            row = coverage[i]
            for j in range(j1, j2):
                row[j] += 1

    def add_delta_x(i, j1, j2):      # add column-strip i over rows [j1,j2)
        cr = coverage[i]; sr = signed[i]; s = 0
        for j in range(j1, j2):
            if cr[j] == 0:
                s -= sr[j]
        return s

    def rem_delta_x(i, j1, j2):      # remove column-strip i over rows [j1,j2)
        cr = coverage[i]; sr = signed[i]; s = 0
        for j in range(j1, j2):
            if cr[j] == 1:
                s += sr[j]
        return s

    def add_delta_y(j, i1, i2):      # add row-strip j over cols [i1,i2)
        s = 0
        for i in range(i1, i2):
            if coverage[i][j] == 0:
                s -= signed[i][j]
        return s

    def rem_delta_y(j, i1, i2):
        s = 0
        for i in range(i1, i2):
            if coverage[i][j] == 1:
                s += signed[i][j]
        return s

    def toggle(i1, i2, j1, j2, delta):  # delta=+1 add, -1 remove
        for i in range(i1, i2):
            row = coverage[i]
            for j in range(j1, j2):
                row[j] += delta

    improved = True
    iters = 0
    while improved and iters < 6:
        improved = False; iters += 1
        if time.monotonic() - _T0 > TIME_LIMIT:
            break
        for idx in range(len(current)):
            if out_bounds[idx] is None or time.monotonic() - _T0 > TIME_LIMIT:
                continue
            i1, i2, j1, j2 = out_bounds[idx]

            # LEFT edge: best new left position p (grow p<i1, shrink p>i1).
            best_d = 0; best_p = i1; acc = 0
            for p in range(i1 - 1, -1, -1):       # grow left
                acc += add_delta_x(p, j1, j2)
                if acc < best_d:
                    best_d = acc; best_p = p
            acc = 0
            for p in range(i1 + 1, i2):           # shrink left
                acc += rem_delta_x(p - 1, j1, j2)
                if acc < best_d:
                    best_d = acc; best_p = p
            if best_p < i1:
                toggle(best_p, i1, j1, j2, 1)
            elif best_p > i1:
                toggle(i1, best_p, j1, j2, -1)
            if best_p != i1:
                i1 = best_p; improved = True

            # RIGHT edge: best new right position q (grow q>i2, shrink q<i2).
            best_d = 0; best_q = i2; acc = 0
            for q in range(i2 + 1, nx + 1):       # grow right
                acc += add_delta_x(q - 1, j1, j2)
                if acc < best_d:
                    best_d = acc; best_q = q
            acc = 0
            for q in range(i2 - 1, i1, -1):       # shrink right
                acc += rem_delta_x(q, j1, j2)
                if acc < best_d:
                    best_d = acc; best_q = q
            if best_q > i2:
                toggle(i2, best_q, j1, j2, 1)
            elif best_q < i2:
                toggle(best_q, i2, j1, j2, -1)
            if best_q != i2:
                i2 = best_q; improved = True

            # TOP edge (j1).
            best_d = 0; best_p = j1; acc = 0
            for p in range(j1 - 1, -1, -1):       # grow down
                acc += add_delta_y(p, i1, i2)
                if acc < best_d:
                    best_d = acc; best_p = p
            acc = 0
            for p in range(j1 + 1, j2):           # shrink
                acc += rem_delta_y(p - 1, i1, i2)
                if acc < best_d:
                    best_d = acc; best_p = p
            if best_p < j1:
                toggle(i1, i2, best_p, j1, 1)
            elif best_p > j1:
                toggle(i1, i2, j1, best_p, -1)
            if best_p != j1:
                j1 = best_p; improved = True

            # BOTTOM edge (j2).
            best_d = 0; best_q = j2; acc = 0
            for q in range(j2 + 1, ny + 1):       # grow up
                acc += add_delta_y(q - 1, i1, i2)
                if acc < best_d:
                    best_d = acc; best_q = q
            acc = 0
            for q in range(j2 - 1, j1, -1):       # shrink
                acc += rem_delta_y(q, i1, i2)
                if acc < best_d:
                    best_d = acc; best_q = q
            if best_q > j2:
                toggle(i1, i2, j2, best_q, 1)
            elif best_q < j2:
                toggle(i1, i2, best_q, j2, -1)
            if best_q != j2:
                j2 = best_q; improved = True

            out_bounds[idx] = (i1, i2, j1, j2)
            current[idx] = (xs[i1], ys[j1], xs[i2], ys[j2])

    return current


def drop_down_cluster(cluster_rects, k):
    """
    Greedy drop-down: start with all n input rects, repeatedly remove the rect
    with the lowest removal cost (unique union area it covers exclusively) until
    k remain. No pair merges — this keeps it fast for any n.
    """
    n = len(cluster_rects)
    if k >= n:
        return list(cluster_rects)

    xs_set = set(); ys_set = set()
    for r in cluster_rects:
        xs_set.add(r[0]); xs_set.add(r[2])
        ys_set.add(r[1]); ys_set.add(r[3])
    xs = sorted(xs_set); ys = sorted(ys_set)
    nx = len(xs) - 1; ny = len(ys) - 1
    if nx <= 0 or ny <= 0:
        return list(cluster_rects[:k])

    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    if nx * ny > CELL_CAP or _grid_work(cluster_rects, xi, yi) > WORK_CAP:
        return _largest_rects_fallback(cluster_rects, k)[0]
    cell_dx = [xs[i + 1] - xs[i] for i in range(nx)]
    cell_dy = [ys[j + 1] - ys[j] for j in range(ny)]

    A_grid = [[False] * ny for _ in range(nx)]
    for r in cluster_rects:
        i1, i2, j1, j2 = xi[r[0]], xi[r[2]], yi[r[1]], yi[r[3]]
        for i in range(i1, i2):
            row = A_grid[i]
            for j in range(j1, j2):
                row[j] = True

    signed = [[0] * ny for _ in range(nx)]
    for i in range(nx):
        dx = cell_dx[i]; row_a = A_grid[i]; row_s = signed[i]
        for j in range(ny):
            a = dx * cell_dy[j]
            row_s[j] = a if row_a[j] else -a

    coverage = [[0] * ny for _ in range(nx)]
    current = list(cluster_rects)
    bounds = []
    for r in current:
        b = (xi[r[0]], xi[r[2]], yi[r[1]], yi[r[3]])
        bounds.append(b)
        i1, i2, j1, j2 = b
        for i in range(i1, i2):
            row = coverage[i]
            for j in range(j1, j2):
                row[j] += 1

    def removal_cost(b):
        i1, i2, j1, j2 = b; s = 0
        for i in range(i1, i2):
            cr = coverage[i]; sr = signed[i]
            for j in range(j1, j2):
                if cr[j] == 1:
                    s += sr[j]
        return s

    costs = [removal_cost(b) for b in bounds]
    dirty = [False] * n

    while len(current) > k:
        if time.monotonic() - _T0 > TIME_LIMIT:
            break

        for idx in range(len(current)):
            if dirty[idx]:
                costs[idx] = removal_cost(bounds[idx])
                dirty[idx] = False

        best_idx = min(range(len(current)), key=lambda i: costs[i])
        b = bounds[best_idx]
        i1, i2, j1, j2 = b
        for i in range(i1, i2):
            row = coverage[i]
            for j in range(j1, j2):
                row[j] -= 1

        for idx in range(len(current)):
            if idx == best_idx or dirty[idx]:
                continue
            bi1, bi2, bj1, bj2 = bounds[idx]
            if bi1 < i2 and i1 < bi2 and bj1 < j2 and j1 < bj2:
                dirty[idx] = True

        current.pop(best_idx)
        bounds.pop(best_idx)
        costs.pop(best_idx)
        dirty.pop(best_idx)

    return current


def kmeans_cluster_solve(cluster_rects, k):
    """Group rects into k spatial groups by centers, return bbox of each group."""
    n = len(cluster_rects)
    if k >= n:
        out = list(cluster_rects[:k])
        while len(out) < k:
            out.append(cluster_rects[0])
        return out
    if k <= 0:
        return [cluster_rects[0]]
    centers = [((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0) for r in cluster_rects]
    sizes = [(r[2] - r[0]) * (r[3] - r[1]) for r in cluster_rects]
    rng = random.Random(12345)
    chosen = [rng.randrange(n)]
    dist2 = [float('inf')] * n
    for _ in range(k - 1):
        cx, cy = centers[chosen[-1]]
        for i in range(n):
            dx = centers[i][0] - cx; dy = centers[i][1] - cy
            d = dx * dx + dy * dy
            if d < dist2[i]: dist2[i] = d
        total_w = sum(dist2[i] * sizes[i] for i in range(n))
        if total_w <= 0:
            chosen.append(rng.randrange(n))
        else:
            r_val = rng.random() * total_w
            acc = 0.0; pick = n - 1
            for i in range(n):
                acc += dist2[i] * sizes[i]
                if acc >= r_val: pick = i; break
            chosen.append(pick)
    centroids = [centers[c] for c in chosen]
    labels = [0] * n
    for _ in range(15):
        changed = False
        for i in range(n):
            cx, cy = centers[i]
            best = 0; bd = float('inf')
            for ci in range(k):
                ccx, ccy = centroids[ci]
                d = (cx - ccx) ** 2 + (cy - ccy) ** 2
                if d < bd: bd = d; best = ci
            if labels[i] != best: labels[i] = best; changed = True
        if not changed: break
        sx = [0.0] * k; sy = [0.0] * k; sw = [0.0] * k
        for i in range(n):
            lb = labels[i]; w = sizes[i]
            sx[lb] += centers[i][0] * w; sy[lb] += centers[i][1] * w; sw[lb] += w
        for ci in range(k):
            if sw[ci] > 0:
                centroids[ci] = (sx[ci] / sw[ci], sy[ci] / sw[ci])
    groups = [[] for _ in range(k)]
    for i, lb in enumerate(labels):
        groups[lb].append(cluster_rects[i])
    out = []
    for g in groups:
        if not g: continue
        x1 = min(r[0] for r in g); y1 = min(r[1] for r in g)
        x2 = max(r[2] for r in g); y2 = max(r[3] for r in g)
        out.append((x1, y1, x2, y2))
    while len(out) < k:
        out.append(out[0] if out else cluster_rects[0])
    return out[:k]


def score_against_target(out_rects, ref_rects):
    """Compute symmetric difference area between unions of the two rect lists."""
    xs_set = set(); ys_set = set()
    for r in list(ref_rects) + list(out_rects):
        xs_set.add(r[0]); xs_set.add(r[2])
        ys_set.add(r[1]); ys_set.add(r[3])
    xs = sorted(xs_set); ys = sorted(ys_set)
    nx = len(xs) - 1; ny = len(ys) - 1
    if nx <= 0 or ny <= 0:
        return 0
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    if (nx * ny > CELL_CAP or _grid_work(ref_rects, xi, yi) > WORK_CAP
            or _grid_work(out_rects, xi, yi) > WORK_CAP):
        # Too large to rasterise safely; report +inf so callers never prefer
        # this branch over an already-computed cheap result.
        return float('inf')
    A = [[False] * ny for _ in range(nx)]
    B = [[False] * ny for _ in range(nx)]
    for r in ref_rects:
        i1, i2, j1, j2 = xi[r[0]], xi[r[2]], yi[r[1]], yi[r[3]]
        for i in range(i1, i2):
            row = A[i]
            for j in range(j1, j2): row[j] = True
    for r in out_rects:
        i1, i2, j1, j2 = xi[r[0]], xi[r[2]], yi[r[1]], yi[r[3]]
        for i in range(i1, i2):
            row = B[i]
            for j in range(j1, j2): row[j] = True
    s = 0
    for i in range(nx):
        dx = xs[i + 1] - xs[i]
        ar = A[i]; br = B[i]
        for j in range(ny):
            if ar[j] != br[j]:
                s += dx * (ys[j + 1] - ys[j])
    return s


def main():
    _N, K, rects = read_input()

    # K >= N: output every input rect (exact reproduction of A, perfect score).
    if K >= _N:
        out = list(rects)
        while len(out) < K:
            out.append(rects[0])
        sys.stdout.write('\n'.join(f"{r[0]} {r[1]} {r[2]} {r[3]}" for r in out[:K]) + '\n')
        return

    # Check if global approach (all coordinate combinations) is feasible
    xs_g = set(); ys_g = set()
    for r in rects:
        xs_g.add(r[0]); xs_g.add(r[2])
        ys_g.add(r[1]); ys_g.add(r[3])
    nx_g = len(xs_g) - 1; ny_g = len(ys_g) - 1
    n_xp_g = nx_g * (nx_g + 1) // 2
    n_yp_g = ny_g * (ny_g + 1) // 2
    global_feasible = n_xp_g * n_yp_g <= 700000

    if global_feasible:
        # Global build-up: enumerate all coord combos across the entire input
        bu_out, _ = build_up(rects, K, all_combos=True)
        bu_out = optimize_edges(bu_out, rects)
        while len(bu_out) < K:
            bu_out.append(bu_out[0] if bu_out else rects[0])
        out_rects = bu_out

        # Also try drop-down (better when K/N is high and individual rects are good)
        if _N > K and time.monotonic() - _T0 < TIME_LIMIT - 0.3:
            dd_out = drop_down_cluster(rects, K)
            while len(dd_out) < K:
                dd_out.append(dd_out[0] if dd_out else rects[0])
            if time.monotonic() - _T0 < TIME_LIMIT - 0.2:
                dd_out = optimize_edges(dd_out, rects)
            if score_against_target(dd_out, rects) < score_against_target(out_rects, rects):
                out_rects = dd_out

        sys.stdout.write('\n'.join(f"{r[0]} {r[1]} {r[2]} {r[3]}" for r in out_rects[:K]) + '\n')
        return

    # Per-cluster approach for larger inputs
    clusters = find_clusters(rects)
    C = len(clusters)
    cluster_rects_list = [[rects[i] for i in c] for c in clusters]
    sizes = [len(c) for c in cluster_rects_list]

    all_rects_seq = []
    all_gains_seq = []

    for cr in cluster_rects_list:
        n_i = len(cr)
        max_b = min(n_i, K)
        if time.monotonic() - _T0 > TIME_LIMIT:
            r_out = kmeans_cluster_solve(cr, max_b)
            all_rects_seq.append(r_out)
            all_gains_seq.append([1] + [0] * (max_b - 1))
        else:
            r_out, g_out = build_up(cr, max_b)
            all_rects_seq.append(r_out)
            all_gains_seq.append(g_out)

    # Allocate all K budget by marginal gain, FROM SCRATCH (every cluster starts
    # at 0). Each added rect's gain == reduction in |A△B|, so popping the highest
    # marginal gain greedily minimises symmetric difference. Allocating from 0
    # (rather than forcing 1 per cluster) lets a big cluster's 2nd/3rd rect beat a
    # tiny cluster's 1st rect — critical when K≈C and big clusters are starved.
    budgets = [0] * C
    heap = []
    for i in range(C):
        gs = all_gains_seq[i]
        if len(gs) > 0 and gs[0] > 0:
            heapq.heappush(heap, (-gs[0], i, 0))
    for _ in range(K):
        if not heap:
            break
        _, i, pos = heapq.heappop(heap)
        budgets[i] += 1
        nxt = pos + 1
        gs = all_gains_seq[i]
        if nxt < len(gs) and gs[nxt] > 0 and budgets[i] < sizes[i]:
            heapq.heappush(heap, (-gs[nxt], i, nxt))

    out = []
    # Process biggest-budget clusters first: they hold most of the uncovered
    # area and benefit most from the time-bounded refinement pass. The remaining
    # time budget is shared fairly across clusters in proportion to their budget.
    order = sorted(range(C), key=lambda i: budgets[i], reverse=True)
    remaining_b = sum(budgets[i] for i in order)
    for i in order:
        cr = cluster_rects_list[i]
        b = budgets[i]
        if b <= 0:
            continue
        # Build-up result
        bu_out = list(all_rects_seq[i][:b])
        while len(bu_out) < b:
            bu_out.append(cr[0])

        # Drop-down result (fast — no pair merges)
        cluster_out = bu_out
        if time.monotonic() - _T0 < TIME_LIMIT - 0.3:
            dd_out = drop_down_cluster(cr, b)
            while len(dd_out) < b:
                dd_out.append(cr[0])
            if score_against_target(dd_out, cr) < score_against_target(bu_out, cr):
                cluster_out = dd_out

        # Time-bounded coordinate descent: alternate edge sliding (grow gaps /
        # trim excess) with relocation of redundant rects. Each step is monotone
        # in score, so iterating until a fixed point or the time slice runs out
        # only helps.
        time_left = (_T0 + TIME_LIMIT - 0.15) - time.monotonic()
        if b >= 2 and time_left > 0.02 and remaining_b > 0:
            slice_dl = time.monotonic() + time_left * (b / remaining_b)
            prev_sc = None
            for _round in range(6):
                cluster_out = optimize_edges(cluster_out, cr)
                if time.monotonic() >= slice_dl:
                    break
                cluster_out = refine_cluster(cr, cluster_out, slice_dl)
                sc = score_against_target(cluster_out, cr)
                if prev_sc is not None and sc >= prev_sc:
                    cluster_out = optimize_edges(cluster_out, cr)
                    break
                prev_sc = sc
                if time.monotonic() >= slice_dl:
                    break
        elif time.monotonic() - _T0 < TIME_LIMIT - 0.1:
            cluster_out = optimize_edges(cluster_out, cr)

        remaining_b -= b
        out.extend(cluster_out)

    while len(out) < K:
        out.append(out[0] if out else (0, 0, 1, 1))
    out = out[:K]

    sys.stdout.write('\n'.join(f"{r[0]} {r[1]} {r[2]} {r[3]}" for r in out) + '\n')


if __name__ == '__main__':
    main()
