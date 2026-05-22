import sys
from bisect import bisect_left
from collections import defaultdict

def solve():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    B = int(data[idx]); idx += 1
    alpha = int(data[idx]); idx += 1
    beta = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]

    if n == 0:
        print(0)
        return

    P = [0] * (n + 1)
    for i in range(n):
        P[i + 1] = P[i] + a[i]

    _fb_cache = {}
    def full_blocks(s):
        """All (l,r) with sum s. Sorted (r ASC, l DESC) — prefer shorter blocks."""
        if s in _fb_cache:
            return _fb_cache[s]
        out = []
        for L in range(1, n + 1):
            target = P[L - 1] + s
            r = bisect_left(P, target, L, n + 1)
            if r <= n and P[r] == target:
                out.append((r, L))
        out.sort(key=lambda x: (x[0], -x[1]))
        _fb_cache[s] = out
        return out

    def greedy_count(s):
        """Max non-overlapping blocks for sum s. O(n)."""
        cnt = 0
        seen = {P[0]: 0}
        for r in range(1, n + 1):
            need = P[r] - s
            if need in seen:
                cnt += 1; seen = {P[r]: r}
            else:
                if P[r] not in seen:
                    seen[P[r]] = r
        return cnt

    def merge(A, Bl):
        """Merge two sorted (r asc, l desc) block lists. O(|A|+|Bl|)."""
        res = []; i = j = 0; la, lb = len(A), len(Bl)
        while i < la and j < lb:
            ra, la_ = A[i]; rb, lb_ = Bl[j]
            if ra < rb or (ra == rb and la_ >= lb_):
                res.append(A[i]); i += 1
            else:
                res.append(Bl[j]); j += 1
        if i < la: res.extend(A[i:])
        if j < lb: res.extend(Bl[j:])
        return res

    def count_select(blocks):
        cnt = last = 0
        for r, l in blocks:
            if l > last:
                cnt += 1; last = r
        return cnt

    def evaluate(sum_list):
        """Activity select with color tracking. Returns (obj, [(l,r,s)])."""
        tagged = []
        for s in sum_list:
            for r, l in full_blocks(s):
                tagged.append((r, l, s))
        tagged.sort(key=lambda x: (x[0], -x[1]))
        chosen = []; used = set(); last = 0
        for r, l, s in tagged:
            if l > last:
                chosen.append((l, r, s)); used.add(s); last = r
        C = len(used)
        return (alpha * len(chosen) - beta * C if C else 0), chosen

    freq = [0] * 51
    for v in a:
        freq[v] += 1

    best_obj = 0
    best_sol = []

    def try_update(obj, chosen):
        nonlocal best_obj, best_sol
        if obj > best_obj:
            best_obj = obj; best_sol = list(chosen)

    # --- Candidate generation ---
    # All single-element values present in array.
    cands = set(v for v in range(51) if freq[v] > 0)

    # Multi-element sums from windows of size 2..20.
    mfreq = defaultdict(int)
    for start in range(n):
        s = 0
        for k in range(min(20, n - start)):
            s += a[start + k]
            if k >= 1:
                mfreq[s] += 1

    thr = max(2, n // 100)
    for s, c in mfreq.items():
        if c >= thr:
            cands.add(s)

    # Rank by greedy count; keep top candidates.
    gcnt = {s: greedy_count(s) for s in cands}
    sorted_cands = sorted(cands, key=lambda s: -gcnt[s])[:max(70, B * 6)]

    # Precompute block lists for all candidates.
    for s in sorted_cands:
        full_blocks(s)

    # --- Phase 1: Quick single-element optimal (exact, no activity select needed) ---
    # OBJ = alpha*sum(freq[v]) - beta*C is exact since single-element blocks never overlap.
    se_vals = sorted([v for v in range(51) if freq[v] > 0], key=lambda v: -freq[v])
    se_set = []
    for v in se_vals:
        if len(se_set) >= B:
            break
        if alpha * freq[v] > beta or not se_set:
            se_set.append(v)
        else:
            break
    if not se_set and se_vals:
        se_set = [se_vals[0]]

    # Evaluate all prefix lengths with full blocks (captures multi-element bonus).
    fm0 = []
    for i, v in enumerate(se_set):
        fm0 = merge(fm0, full_blocks(v))
        obj, chosen = evaluate(se_set[:i + 1])
        try_update(obj, chosen)

    # --- Phase 2: Greedy forward pass over all candidates ---
    # Uses incremental merge (O(k) per trial), calls evaluate() once per iteration.
    cur_set = []
    cur_merged = []
    cur_obj = 0
    cur_C = 0

    for _ in range(B):
        best_gain = 0
        bs = None; bm = None

        for s in sorted_cands:
            if s in cur_set:
                continue
            trial = merge(cur_merged, full_blocks(s))
            cnt = count_select(trial)
            obj_est = alpha * cnt - beta * (cur_C + 1)
            gain = obj_est - cur_obj
            if gain > best_gain:
                best_gain = gain; bs = s; bm = trial

        if bs is None:
            break

        cur_set.append(bs)
        cur_merged = bm

        obj, chosen = evaluate(cur_set)
        cur_obj = obj
        cur_C = len(set(s for _, _, s in chosen)) if chosen else 0
        try_update(obj, chosen)

    # --- Phase 3: Swap refinement ---
    # For each sum in cur_set, try replacing it with a better candidate.
    # Uses prefix/suffix precomputed merges so each trial is O(k), not O(B*k).
    # Guard: skip if merged list is too large (would be slow in Python).
    if cur_set and len(cur_merged) < 40000:
        for _ in range(3):
            swapped = False

            # Precompute prefix[i] = merged blocks for cur_set[0..i-1]
            pre = [[]]
            for s in cur_set:
                pre.append(merge(pre[-1], full_blocks(s)))

            # Precompute suffix[i] = merged blocks for cur_set[i..end]
            suf = [None] * (len(cur_set) + 1)
            suf[len(cur_set)] = []
            for i in range(len(cur_set) - 1, -1, -1):
                suf[i] = merge(full_blocks(cur_set[i]), suf[i + 1])

            for i in range(len(cur_set)):
                partial = merge(pre[i], suf[i + 1])

                best_swap_gain = 0
                best_swap_s = None

                for s_new in sorted_cands:
                    if s_new in cur_set:
                        continue
                    trial = merge(partial, full_blocks(s_new))
                    cnt = count_select(trial)
                    # Assume C stays same (swap, not add).
                    obj_est = alpha * cnt - beta * cur_C
                    if obj_est - best_obj > best_swap_gain:
                        best_swap_gain = obj_est - best_obj
                        best_swap_s = s_new
                        best_swap_m = trial

                if best_swap_s is not None:
                    new_set = cur_set[:i] + [best_swap_s] + cur_set[i + 1:]
                    obj_ex, chosen = evaluate(new_set)
                    if obj_ex > best_obj:
                        try_update(obj_ex, chosen)
                        cur_set = new_set
                        cur_merged = best_swap_m
                        cur_C = len(set(s for _, _, s in chosen)) if chosen else 0
                        cur_obj = obj_ex
                        swapped = True
                        break

            if not swapped:
                break

    # --- Output ---
    if best_obj <= 0 or not best_sol:
        print(0)
        return

    used_sums = sorted(set(s for _, _, s in best_sol))
    s2c = {s: i + 1 for i, s in enumerate(used_sums)}
    out = [str(len(best_sol))]
    for l, r, s in best_sol:
        out.append(f"{l} {r} {s2c[s]}")
    sys.stdout.write('\n'.join(out) + '\n')

solve()
