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


def build_up(input_rects, k, all_combos=False):
    """
    Greedy build-up: iteratively add the best candidate rect.

    gain(R) = area(R ∩ Union NOT yet covered) − area(R NOT in Union NOT yet covered).

    Candidate rects depend on grid size:
      - all_combos=True or total_combos <= AUTO_THRESH:
          enumerate ALL (i1,i2,j1,j2) using 2D prefix sums (O(1) query).
      - else: restricted candidate set with direct queries.

    Returns (output_rects, marginal_gains), both length k.
    """
    AUTO_THRESH = 600000
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
    cell_dx = [xs[i + 1] - xs[i] for i in range(nx)]
    cell_dy = [ys[j + 1] - ys[j] for j in range(ny)]

    # --- Build union grid and val grid ---
    union = [[False] * ny for _ in range(nx)]
    for r in input_rects:
        i1, i2, j1, j2 = xi[r[0]], xi[r[2]], yi[r[1]], yi[r[3]]
        for i in range(i1, i2):
            row = union[i]
            for j in range(j1, j2):
                row[j] = True

    val = [[0] * ny for _ in range(nx)]
    for i in range(nx):
        dx = cell_dx[i]
        row_u = union[i]; row_v = val[i]
        for j in range(ny):
            a = dx * cell_dy[j]
            row_v[j] = a if row_u[j] else -a

    # --- Candidate generation ---
    n_xp = nx * (nx + 1) // 2
    n_yp = ny * (ny + 1) // 2
    total = n_xp * n_yp

    use_all_combos = all_combos or total <= AUTO_THRESH

    if use_all_combos:
        cands = [(i1, i2, j1, j2)
                 for i1 in range(nx) for i2 in range(i1 + 1, nx + 1)
                 for j1 in range(ny) for j2 in range(j1 + 1, ny + 1)]
    else:
        cand_set = set()
        for r in input_rects:
            cand_set.add((xi[r[0]], xi[r[2]], yi[r[1]], yi[r[3]]))
        # All pairs for small-medium clusters, sliding window for large
        if n <= 150:
            cr = input_rects
            for i in range(n):
                ri = cr[i]
                for j in range(i + 1, n):
                    rj = cr[j]
                    cand_set.add((
                        min(xi[ri[0]], xi[rj[0]]), max(xi[ri[2]], xi[rj[2]]),
                        min(yi[ri[1]], yi[rj[1]]), max(yi[ri[3]], yi[rj[3]])
                    ))
        else:
            W = 30
            order = sorted(range(n), key=lambda idx: input_rects[idx][0] + input_rects[idx][2])
            cr = input_rects
            for oi in range(n):
                ri = cr[order[oi]]
                for oj in range(oi + 1, min(n, oi + 1 + W)):
                    rj = cr[order[oj]]
                    cand_set.add((
                        min(xi[ri[0]], xi[rj[0]]), max(xi[ri[2]], xi[rj[2]]),
                        min(yi[ri[1]], yi[rj[1]]), max(yi[ri[3]], yi[rj[3]])
                    ))
            # Also do y-sorted window for better vertical merges
            order_y = sorted(range(n), key=lambda idx: input_rects[idx][1] + input_rects[idx][3])
            for oi in range(n):
                ri = cr[order_y[oi]]
                for oj in range(oi + 1, min(n, oi + 1 + W)):
                    rj = cr[order_y[oj]]
                    cand_set.add((
                        min(xi[ri[0]], xi[rj[0]]), max(xi[ri[2]], xi[rj[2]]),
                        min(yi[ri[1]], yi[rj[1]]), max(yi[ri[3]], yi[rj[3]])
                    ))
        cands = list(cand_set)

    # --- Build-up loop (always uses prefix sums for O(1) per-candidate queries) ---
    output_rects = []
    gains = []

    for _ in range(k):
        if time.monotonic() - _T0 > TIME_LIMIT:
            break

        best_g = 0
        best_c = None

        # Rebuild 2D prefix sum
        psum = [[0] * (ny + 1) for _ in range(nx + 1)]
        for i in range(1, nx + 1):
            rp = psum[i]; rpm = psum[i - 1]
            rv = val[i - 1]
            for j in range(1, ny + 1):
                rp[j] = rv[j - 1] + rpm[j] + rp[j - 1] - rpm[j - 1]

        for (i1, i2, j1, j2) in cands:
            g = psum[i2][j2] - psum[i1][j2] - psum[i2][j1] + psum[i1][j1]
            if g > best_g:
                best_g = g; best_c = (i1, i2, j1, j2)

        if best_c is None:
            break

        gains.append(best_g)
        i1, i2, j1, j2 = best_c
        output_rects.append((xs[i1], ys[j1], xs[i2], ys[j2]))

        for i in range(i1, i2):
            rv = val[i]
            for j in range(j1, j2):
                rv[j] = 0

    while len(output_rects) < k:
        output_rects.append(output_rects[0] if output_rects else input_rects[0])
        gains.append(0)

    return output_rects, gains


def shrink_output(output_rects, ref_rects):
    """Shrink each output rect inward when removing an edge strip reduces symdiff."""
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

    def shrink_delta(i1, i2, j1, j2):
        s = 0
        for i in range(i1, i2):
            cr_ = coverage[i]; sr = signed[i]
            for j in range(j1, j2):
                if cr_[j] == 1:
                    s += sr[j]
        return s

    improved = True
    iters = 0
    while improved and iters < 4:
        improved = False; iters += 1
        for idx in range(len(current)):
            if out_bounds[idx] is None or time.monotonic() - _T0 > TIME_LIMIT:
                continue
            i1, i2, j1, j2 = out_bounds[idx]
            r = current[idx]
            best_new = None; best_delta = 0
            for t in range(1, i2 - i1):
                d = shrink_delta(i1, i1 + t, j1, j2)
                if d < best_delta:
                    best_delta = d; best_new = (xs[i1 + t], r[1], r[2], r[3])
            for t in range(1, i2 - i1):
                d = shrink_delta(i2 - t, i2, j1, j2)
                if d < best_delta:
                    best_delta = d; best_new = (r[0], r[1], xs[i2 - t], r[3])
            for t in range(1, j2 - j1):
                d = shrink_delta(i1, i2, j1, j1 + t)
                if d < best_delta:
                    best_delta = d; best_new = (r[0], ys[j1 + t], r[2], r[3])
            for t in range(1, j2 - j1):
                d = shrink_delta(i1, i2, j2 - t, j2)
                if d < best_delta:
                    best_delta = d; best_new = (r[0], r[1], r[2], ys[j2 - t])
            if best_new is not None:
                for i in range(i1, i2):
                    row = coverage[i]
                    for j in range(j1, j2):
                        row[j] -= 1
                ni1 = xi.get(best_new[0], i1)
                ni2 = xi.get(best_new[2], i2)
                nj1 = yi.get(best_new[1], j1)
                nj2 = yi.get(best_new[3], j2)
                out_bounds[idx] = (ni1, ni2, nj1, nj2)
                for i in range(ni1, ni2):
                    row = coverage[i]
                    for j in range(nj1, nj2):
                        row[j] += 1
                current[idx] = best_new
                improved = True

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
    A = [[False] * ny for _ in range(nx)]
    B = [[False] * ny for _ in range(nx)]
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
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
        bu_out = shrink_output(bu_out, rects)
        while len(bu_out) < K:
            bu_out.append(bu_out[0] if bu_out else rects[0])
        out_rects = bu_out

        # Also try drop-down (better when K/N is high and individual rects are good)
        if _N > K and time.monotonic() - _T0 < TIME_LIMIT - 0.3:
            dd_out = drop_down_cluster(rects, K)
            while len(dd_out) < K:
                dd_out.append(dd_out[0] if dd_out else rects[0])
            if time.monotonic() - _T0 < TIME_LIMIT - 0.2:
                dd_out = shrink_output(dd_out, rects)
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
    for i, (cr, b) in enumerate(zip(cluster_rects_list, budgets)):
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

        if time.monotonic() - _T0 < TIME_LIMIT - 0.2:
            cluster_out = shrink_output(cluster_out, cr)

        out.extend(cluster_out)

    while len(out) < K:
        out.append(out[0] if out else (0, 0, 1, 1))
    out = out[:K]

    sys.stdout.write('\n'.join(f"{r[0]} {r[1]} {r[2]} {r[3]}" for r in out) + '\n')


if __name__ == '__main__':
    main()
