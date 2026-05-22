#!/usr/bin/env python3
"""
Rectangle Approximation - pure Python (no numpy).

Strategy:
1. Cluster rectangles by transitive interior overlap. K >= C.
2. Distribute K-C extra budget across clusters proportional to size.
3. Per cluster, reduce from n down to k by greedily picking the cheapest of:
   - drop one rectangle, or
   - merge two rectangles into their bounding box.
   Cost = change in symmetric difference, computed on a coordinate-
   compressed grid. Drop costs are cached & invalidated only on overlap.
4. Post-process: shrink each output rect inward as long as it helps.
"""
import sys
import bisect
import random


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


def bbox_union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


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


def approximate_cluster(cluster_rects, k):
    n = len(cluster_rects)
    if k >= n:
        out = list(cluster_rects)
        while len(out) < k:
            out.append(cluster_rects[0])
        return out[:k]

    # Coordinate compression
    xs_set = set(); ys_set = set()
    for r in cluster_rects:
        xs_set.add(r[0]); xs_set.add(r[2])
        ys_set.add(r[1]); ys_set.add(r[3])
    xs = sorted(xs_set); ys = sorted(ys_set)
    nx = len(xs) - 1
    ny = len(ys) - 1

    cell_dx = [xs[i+1] - xs[i] for i in range(nx)]
    cell_dy = [ys[j+1] - ys[j] for j in range(ny)]

    # signed[i][j] = +cell_area if cell in A, else -cell_area.
    # Removing 1->0 coverage on cell contributes +signed to symdiff.
    # Adding 0->1 coverage contributes -signed.
    A_grid = [[False] * ny for _ in range(nx)]
    for r in cluster_rects:
        i1 = bisect.bisect_left(xs, r[0])
        i2 = bisect.bisect_left(xs, r[2])
        j1 = bisect.bisect_left(ys, r[1])
        j2 = bisect.bisect_left(ys, r[3])
        for i in range(i1, i2):
            row = A_grid[i]
            for j in range(j1, j2):
                row[j] = True

    signed = [[0]*ny for _ in range(nx)]
    for i in range(nx):
        dx = cell_dx[i]
        ar = A_grid[i]
        sr = signed[i]
        for j in range(ny):
            a = dx * cell_dy[j]
            sr[j] = a if ar[j] else -a

    # 1D row sums for fast removal_cost when coverage is uniform 1 in big blocks
    # We won't precompute row prefix sums for signed because coverage varies.
    # We'll just iterate cells.

    coverage = [[0] * ny for _ in range(nx)]
    current = list(cluster_rects)

    def cells_of(rect):
        i1 = bisect.bisect_left(xs, rect[0])
        i2 = bisect.bisect_left(xs, rect[2])
        j1 = bisect.bisect_left(ys, rect[1])
        j2 = bisect.bisect_left(ys, rect[3])
        return (i1, i2, j1, j2)

    bounds = [cells_of(r) for r in current]
    for (i1, i2, j1, j2) in bounds:
        for i in range(i1, i2):
            row = coverage[i]
            for j in range(j1, j2):
                row[j] += 1

    def removal_cost(b):
        i1, i2, j1, j2 = b
        s = 0
        for i in range(i1, i2):
            cr = coverage[i]
            sr = signed[i]
            for j in range(j1, j2):
                if cr[j] == 1:
                    s += sr[j]
        return s

    def add_cost(b):
        i1, i2, j1, j2 = b
        s = 0
        for i in range(i1, i2):
            cr = coverage[i]
            sr = signed[i]
            for j in range(j1, j2):
                if cr[j] == 0:
                    s -= sr[j]
        return s

    def add_bounds(b):
        i1, i2, j1, j2 = b
        for i in range(i1, i2):
            row = coverage[i]
            for j in range(j1, j2):
                row[j] += 1

    def remove_bounds(b):
        i1, i2, j1, j2 = b
        for i in range(i1, i2):
            row = coverage[i]
            for j in range(j1, j2):
                row[j] -= 1

    drop_costs = [removal_cost(bounds[i]) for i in range(len(current))]
    dirty = [False] * len(current)

    def invalidate_in_region(b):
        i1, i2, j1, j2 = b
        for idx in range(len(bounds)):
            if dirty[idx]:
                continue
            bi1, bi2, bj1, bj2 = bounds[idx]
            if bi1 < i2 and i1 < bi2 and bj1 < j2 and j1 < bj2:
                dirty[idx] = True

    def refresh():
        for idx in range(len(bounds)):
            if dirty[idx]:
                drop_costs[idx] = removal_cost(bounds[idx])
                dirty[idx] = False

    while len(current) > k:
        refresh()

        # Greedily drop any rect with non-positive removal cost.
        progress = True
        while progress and len(current) > k:
            progress = False
            best_idx = -1
            best_c = 1
            for idx in range(len(current)):
                if dirty[idx]:
                    drop_costs[idx] = removal_cost(bounds[idx])
                    dirty[idx] = False
                if drop_costs[idx] < best_c:
                    best_c = drop_costs[idx]
                    best_idx = idx
            if best_idx >= 0 and best_c <= 0:
                b = bounds[best_idx]
                remove_bounds(b)
                invalidate_in_region(b)
                current.pop(best_idx)
                bounds.pop(best_idx)
                drop_costs.pop(best_idx)
                dirty.pop(best_idx)
                progress = True
        if len(current) <= k:
            break
        m = len(current)

        # Best single-drop fallback
        best_drop_idx = 0
        best_drop_cost = drop_costs[0]
        for i in range(1, m):
            if drop_costs[i] < best_drop_cost:
                best_drop_cost = drop_costs[i]
                best_drop_idx = i
        best_delta = best_drop_cost
        best_action = ('drop', best_drop_idx)

        # Pair-merge candidates by spatial proximity (sort by center-x, window).
        centers_x = [(r[0] + r[2]) for r in current]
        order = sorted(range(m), key=lambda i: centers_x[i])
        if m > 500:
            W = 8
        elif m > 250:
            W = 16
        elif m > 100:
            W = 28
        else:
            W = max(40, m)
        pair_budget = 4000
        tried = 0

        for oi in range(m):
            if tried >= pair_budget:
                break
            i = order[oi]
            ri = current[i]
            ri_a = (ri[2]-ri[0]) * (ri[3]-ri[1])
            bi_b = bounds[i]
            for oj in range(oi+1, min(m, oi+1+W)):
                if tried >= pair_budget:
                    break
                j = order[oj]
                rj = current[j]
                rj_a = (rj[2]-rj[0]) * (rj[3]-rj[1])
                bb = bbox_union(ri, rj)
                bb_a = (bb[2]-bb[0]) * (bb[3]-bb[1])
                if bb_a > 4 * (ri_a + rj_a) + 200:
                    continue
                tried += 1
                bj_b = bounds[j]
                ci = drop_costs[i]
                remove_bounds(bi_b)
                cj = removal_cost(bj_b)
                remove_bounds(bj_b)
                bb_bounds = cells_of(bb)
                ca = add_cost(bb_bounds)
                add_bounds(bj_b)
                add_bounds(bi_b)
                d = ci + cj + ca
                if d < best_delta:
                    best_delta = d
                    best_action = ('merge', i, j, bb, bb_bounds)

        if best_action[0] == 'drop':
            idx = best_action[1]
            b = bounds[idx]
            remove_bounds(b)
            invalidate_in_region(b)
            current.pop(idx)
            bounds.pop(idx)
            drop_costs.pop(idx)
            dirty.pop(idx)
        else:
            _, i, j, bb, bb_bounds = best_action
            if i > j:
                i, j = j, i
            bj_b = bounds[j]; bi_b = bounds[i]
            remove_bounds(bj_b)
            remove_bounds(bi_b)
            invalidate_in_region(bj_b)
            invalidate_in_region(bi_b)
            current.pop(j); current.pop(i)
            bounds.pop(j); bounds.pop(i)
            drop_costs.pop(j); drop_costs.pop(i)
            dirty.pop(j); dirty.pop(i)
            current.append(bb)
            bounds.append(bb_bounds)
            drop_costs.append(0)
            dirty.append(False)
            add_bounds(bb_bounds)
            invalidate_in_region(bb_bounds)
            drop_costs[-1] = removal_cost(bb_bounds)

    # ---- Post-processing: shrink each rect inward when beneficial ----
    def shrink_delta(i1, i2, j1, j2):
        """Cost of removing the cells in this sub-rectangle from coverage."""
        s = 0
        for i in range(i1, i2):
            cr = coverage[i]
            sr = signed[i]
            for j in range(j1, j2):
                if cr[j] == 1:
                    s += sr[j]
        return s

    improved = True
    iters = 0
    while improved and iters < 4:
        improved = False
        iters += 1
        for idx in range(len(current)):
            i1, i2, j1, j2 = bounds[idx]
            r = current[idx]
            best_new = None
            best_delta = 0
            # Shrink left edge
            for t in range(1, i2 - i1):
                d = shrink_delta(i1, i1 + t, j1, j2)
                if d < best_delta:
                    best_delta = d
                    best_new = (xs[i1 + t], r[1], r[2], r[3])
            # Shrink right
            for t in range(1, i2 - i1):
                d = shrink_delta(i2 - t, i2, j1, j2)
                if d < best_delta:
                    best_delta = d
                    best_new = (r[0], r[1], xs[i2 - t], r[3])
            # Shrink bottom
            for t in range(1, j2 - j1):
                d = shrink_delta(i1, i2, j1, j1 + t)
                if d < best_delta:
                    best_delta = d
                    best_new = (r[0], ys[j1 + t], r[2], r[3])
            # Shrink top
            for t in range(1, j2 - j1):
                d = shrink_delta(i1, i2, j2 - t, j2)
                if d < best_delta:
                    best_delta = d
                    best_new = (r[0], r[1], r[2], ys[j2 - t])
            if best_new is not None:
                remove_bounds(bounds[idx])
                new_b = cells_of(best_new)
                add_bounds(new_b)
                current[idx] = best_new
                bounds[idx] = new_b
                improved = True

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
    centers = [((r[0]+r[2]) / 2.0, (r[1]+r[3]) / 2.0) for r in cluster_rects]
    sizes = [(r[2]-r[0]) * (r[3]-r[1]) for r in cluster_rects]
    rng = random.Random(12345)
    # k-means++ init weighted by size
    chosen = [rng.randrange(n)]
    dist2 = [float('inf')] * n
    for _ in range(k - 1):
        cx, cy = centers[chosen[-1]]
        for i in range(n):
            dx = centers[i][0] - cx
            dy = centers[i][1] - cy
            d = dx*dx + dy*dy
            if d < dist2[i]:
                dist2[i] = d
        # pick weighted by dist2 * size
        total_w = 0.0
        for i in range(n):
            total_w += dist2[i] * sizes[i]
        if total_w <= 0:
            chosen.append(rng.randrange(n))
        else:
            r = rng.random() * total_w
            acc = 0.0
            pick = n - 1
            for i in range(n):
                acc += dist2[i] * sizes[i]
                if acc >= r:
                    pick = i
                    break
            chosen.append(pick)
    centroids = [centers[c] for c in chosen]
    labels = [0] * n
    for _ in range(15):
        changed = False
        for i in range(n):
            cx, cy = centers[i]
            best = 0
            bd = float('inf')
            for ci in range(k):
                ccx, ccy = centroids[ci]
                dx = cx - ccx; dy = cy - ccy
                d = dx*dx + dy*dy
                if d < bd:
                    bd = d
                    best = ci
            if labels[i] != best:
                labels[i] = best
                changed = True
        if not changed:
            break
        # Update centroids weighted by size
        sx = [0.0] * k; sy = [0.0] * k; sw = [0.0] * k
        for i in range(n):
            lb = labels[i]
            w = sizes[i]
            sx[lb] += centers[i][0] * w
            sy[lb] += centers[i][1] * w
            sw[lb] += w
        for ci in range(k):
            if sw[ci] > 0:
                centroids[ci] = (sx[ci]/sw[ci], sy[ci]/sw[ci])
    groups = [[] for _ in range(k)]
    for i, lb in enumerate(labels):
        groups[lb].append(cluster_rects[i])
    out = []
    for g in groups:
        if not g:
            continue
        x1 = min(r[0] for r in g); y1 = min(r[1] for r in g)
        x2 = max(r[2] for r in g); y2 = max(r[3] for r in g)
        out.append((x1, y1, x2, y2))
    while len(out) < k:
        out.append(out[0])
    return out[:k]


def score_against_target(out_rects, cluster_rects):
    """Compute symmetric difference between unions of the two rect lists."""
    xs_set = set(); ys_set = set()
    for r in cluster_rects:
        xs_set.add(r[0]); xs_set.add(r[2])
        ys_set.add(r[1]); ys_set.add(r[3])
    for r in out_rects:
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
    for r in cluster_rects:
        i1 = xi[r[0]]; i2 = xi[r[2]]; j1 = yi[r[1]]; j2 = yi[r[3]]
        for i in range(i1, i2):
            row = A[i]
            for j in range(j1, j2):
                row[j] = True
    for r in out_rects:
        i1 = xi[r[0]]; i2 = xi[r[2]]; j1 = yi[r[1]]; j2 = yi[r[3]]
        for i in range(i1, i2):
            row = B[i]
            for j in range(j1, j2):
                row[j] = True
    s = 0
    for i in range(nx):
        dx = xs[i+1] - xs[i]
        ar = A[i]; br = B[i]
        for j in range(ny):
            if ar[j] != br[j]:
                s += dx * (ys[j+1] - ys[j])
    return s


def approximate_cluster_best(cluster_rects, k):
    """Try greedy and (optionally) k-means; return whichever is better."""
    n = len(cluster_rects)
    if k >= n:
        out = list(cluster_rects)
        while len(out) < k:
            out.append(cluster_rects[0])
        return out[:k]
    candidates = [approximate_cluster(cluster_rects, k)]
    if n > 2 * k and n <= 300:
        # k-means is cheap-ish; only run for moderate cluster sizes
        candidates.append(kmeans_cluster_solve(cluster_rects, k))
    best = min(candidates, key=lambda c: score_against_target(c, cluster_rects))
    return best


def main():
    N, K, rects = read_input()
    clusters = find_clusters(rects)
    C = len(clusters)
    cluster_rects_list = [[rects[i] for i in c] for c in clusters]
    sizes = [len(c) for c in cluster_rects_list]

    budgets = [1] * C
    extra = K - C
    if extra > 0:
        total = sum(sizes)
        give = [0] * C
        if total > 0:
            for i in range(C):
                g = (extra * sizes[i]) // total
                g = min(g, sizes[i] - 1)
                give[i] = g
        used = sum(give)
        remaining = extra - used
        order = sorted(range(C), key=lambda i: -sizes[i])
        idx = 0
        safety = 0
        while remaining > 0 and safety < 20 * C + 50:
            i = order[idx % C]
            if budgets[i] + give[i] < sizes[i]:
                give[i] += 1
                remaining -= 1
            idx += 1; safety += 1
            if all(budgets[t] + give[t] >= sizes[t] for t in range(C)):
                break
        for i in range(C):
            budgets[i] += give[i]

    out = []
    for cr, b in zip(cluster_rects_list, budgets):
        out.extend(approximate_cluster_best(cr, b))

    while len(out) < K:
        out.append(out[0] if out else (0, 0, 1, 1))
    out = out[:K]

    sys.stdout.write('\n'.join(f"{r[0]} {r[1]} {r[2]} {r[3]}" for r in out) + '\n')


if __name__ == '__main__':
    main()