import sys
from bisect import bisect_right

def solve():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx+=1
    B = int(data[idx]); idx+=1
    alpha = int(data[idx]); idx+=1
    beta = int(data[idx]); idx+=1
    a = [int(data[idx+i]) for i in range(n)]
    idx += n

    # prefix sums
    P = [0]*(n+1)
    for i in range(n):
        P[i+1] = P[i] + a[i]

    # Step 1: For each candidate sum s, compute the greedy maximum-count
    # non-overlapping decomposition.
    # Greedy: scan from left. Maintain pos = current "free" start.
    # For each l from pos to n, find smallest r >= l with prefix[r]-prefix[l-1] == s.
    # That's hard for arbitrary s. Easier: scan r, for each r, smallest l>=pos with
    # prefix[r]-prefix[l-1]=s, i.e. prefix[l-1] = prefix[r]-s.
    #
    # Standard approach for fixed s: walk i from 1..n, track positions of prefix values
    # since `pos-1`. When we see prefix[i] s.t. prefix[i]-s appeared at some j>=pos-1,
    # we take block (j+1, i), set pos = i+1, reset.
    #
    # We want greedy max-count: take the *earliest* end. So for each r, check if there
    # exists l-1 in [pos-1, r-1] with prefix[l-1] = prefix[r]-s. If yes, pick the
    # smallest such l-1 (or any — any choice yields a valid block, and taking it now
    # is optimal greedy).
    #
    # Implementation: for fixed s, iterate r=1..n. Maintain a set of prefix values seen
    # at indices >= pos-1. When prefix[r]-s is in that set, take block ending at r:
    # find some l-1 with that prefix value; we can use the most recent occurrence
    # for shortest block (doesn't matter for count). After taking, set pos=r+1 and
    # clear the set, then continue.
    #
    # To enumerate sums: candidate sums are differences prefix[r]-prefix[l]. We focus
    # on small block sums first (more blocks). Specifically, single elements give us
    # sums equal to a[i]; pairs/triples give other sums. We'll consider sums that
    # appear "many" times.

    # Build a frequency table of single-element values to seed candidates,
    # plus enumerate small-window sums.
    from collections import defaultdict

    # For each sum s, compute greedy block list.
    def greedy_blocks_for_sum(s):
        # returns list of (l, r) 1-indexed
        blocks = []
        pos_prefix_idx = 0  # we accept prefix indices >= pos_prefix_idx
        # map prefix_value -> list of indices (>= pos_prefix_idx)
        seen = {P[pos_prefix_idx]: pos_prefix_idx}
        r = 1
        while r <= n:
            need = P[r] - s
            if need in seen:
                # take block from (seen[need]+1, r)
                lminus1 = seen[need]
                blocks.append((lminus1+1, r))
                # reset
                pos_prefix_idx = r
                seen = {P[r]: r}
                r += 1
            else:
                if P[r] not in seen:
                    seen[P[r]] = r
                r += 1
        return blocks

    # Generate candidate sums:
    # - all values 0..max possible up to some cap
    # - sums of windows of length 1, 2, 3 (small windows produce small sums and many occurrences)
    candidate_sums = set()
    # single-element values
    for v in a:
        candidate_sums.add(v)
    # 2-windows, 3-windows for variety
    for w in (2, 3, 4, 5):
        if w <= n:
            for i in range(n - w + 1):
                s = P[i+w] - P[i]
                candidate_sums.add(s)
                if len(candidate_sums) > 5000:
                    break
            if len(candidate_sums) > 5000:
                break
    # sum 0 is interesting if there are zeros
    if 0 in a:
        candidate_sums.add(0)

    # If too many, keep small ones (more occurrences likely)
    if len(candidate_sums) > 2000:
        candidate_sums = set(sorted(candidate_sums)[:2000])

    # For each candidate sum, run greedy and store block list.
    sum_blocks = {}
    for s in candidate_sums:
        bl = greedy_blocks_for_sum(s)
        if len(bl) >= 1:
            sum_blocks[s] = bl

    # Score a sum alone: if we picked only this color, we'd get alpha*|bl| - beta.
    # We want sums where alpha*|bl| - beta > 0 (otherwise not worth using a color).
    # Sort sums by descending |bl|.
    sorted_sums = sorted(sum_blocks.items(), key=lambda x: -len(x[1]))

    # Try selecting top-K sums (K from 1..B) and run weighted interval scheduling
    # to pick a globally non-overlapping subset, then evaluate objective.

    # Weighted interval scheduling for max number of blocks given a candidate
    # block list (each block has weight 1 for the count maximization).
    # We'll do a more nuanced DP that also limits distinct colors used.
    #
    # Simpler: pick a fixed set S of sums (size <= B). Among union of their block
    # lists, find max-cardinality non-overlapping selection. That's classic: sort
    # by r, greedy-pick earliest r that doesn't conflict.
    #
    # But with multiple block options per sum, we need union of all candidate
    # blocks across sums in S, then standard activity-selection (max # of pairwise
    # non-overlapping intervals) — this is solved greedily by sorting on r.

    def max_nonoverlap(blocks_with_sum):
        # blocks_with_sum: list of (l, r, s)
        # returns chosen list (l,r,s) maximizing count
        if not blocks_with_sum:
            return []
        sorted_b = sorted(blocks_with_sum, key=lambda x: (x[1], x[0]))
        chosen = []
        last_r = 0
        for l, r, s in sorted_b:
            if l > last_r:
                chosen.append((l, r, s))
                last_r = r
        return chosen

    # Generate candidate block pool per sum. The greedy_blocks_for_sum already
    # gives a max greedy set, but those blocks are tuned to *that* sum alone.
    # For combining, we may want a richer pool: all maximal blocks summing to s
    # but starting at various positions. Building all such intervals is up to
    # O(n) per sum (since for each l, there's at most one minimal r reaching sum s
    # when a[i]>=0 monotone... but with zeros it's more). For simplicity,
    # for each sum we'll generate ALL (l,r) with that sum and small width via
    # two-pointer (since a[i]>=0, prefix sums non-decreasing, so for each l there
    # is a unique r-range producing each sum).

    def all_blocks_for_sum(s):
        # since a[i]>=0, prefix is non-decreasing.
        # For each l (1..n), find r such that prefix[r]-prefix[l-1] = s.
        # Since prefix is non-decreasing, we can binary search.
        out = []
        for L in range(1, n+1):
            target = P[L-1] + s
            # find any r in [L, n] with P[r] == target
            # use bisect on P[L..n]
            lo = L
            hi = n
            # binary search for leftmost r >= L with P[r] >= target
            left = lo
            right = hi+1
            while left < right:
                mid = (left+right)//2
                if P[mid] >= target:
                    right = mid
                else:
                    left = mid+1
            r = left
            if r <= n and P[r] == target:
                out.append((L, r))
                # also include subsequent r's with P[r]==target (zeros after)
                rr = r+1
                while rr <= n and P[rr] == target:
                    out.append((L, rr))
                    rr += 1
        return out

    # We'll build pool per chosen sum lazily.

    best_obj = 0
    best_output = []  # list of (l,r,c)

    # Try strategies:
    # Strategy A: For each top sum individually (one color), compute count and obj.
    # Strategy B: Combine top sums up to B in greedy fashion.

    # First gather sums that individually give positive contribution: alpha*cnt - beta > 0
    # i.e. cnt > beta/alpha
    threshold = (beta // alpha) + 1 if alpha > 0 else 1
    # actually need cnt*alpha > beta -> cnt > beta/alpha. For positive obj per color.
    if beta == 0:
        threshold = 1

    viable = [(s, bl) for s, bl in sorted_sums if len(bl) >= threshold]
    # keep at most some number for combinatorial search
    viable_top = viable[:max(B*5, 30)]

    # Strategy: pick subset of sums of size <=B from viable_top to maximize total
    # non-overlap count - we'll greedily build.
    #
    # Greedy add: start empty. Repeatedly try adding the sum that best increases
    # objective. Stop when no improvement.

    # Build per-sum block list (full pool) for viable sums. To save memory, only
    # for top candidates.
    pool_per_sum = {}
    top_for_combine = sorted_sums[:max(B*8, 40)]
    for s, _ in top_for_combine:
        pool_per_sum[s] = all_blocks_for_sum(s)

    def evaluate_set(sum_set):
        # union all candidate blocks, find max non-overlap count.
        # IMPORTANT: a chosen block must "use" exactly one color. After picking
        # max non-overlap, we need each color present at least once to count
        # toward C. We can drop a color entirely (don't use it) if it ended up
        # contributing 0 blocks — but max_nonoverlap may pick blocks from any sum
        # in the set, so 'colors used' = number of distinct sums actually chosen.
        all_b = []
        for s in sum_set:
            for (l,r) in pool_per_sum.get(s, []):
                all_b.append((l, r, s))
        chosen = max_nonoverlap(all_b)
        used_sums = set(s for _,_,s in chosen)
        k = len(chosen)
        C = len(used_sums)
        obj = alpha*k - beta*C
        return obj, chosen

    # Greedy build from empty
    current_set = set()
    cur_obj = 0
    cur_chosen = []

    # First: try each single sum
    candidate_pool = [s for s,_ in top_for_combine]

    improved = True
    while improved and len(current_set) < B:
        improved = False
        best_add = None
        best_new_obj = cur_obj
        best_new_chosen = cur_chosen
        for s in candidate_pool:
            if s in current_set:
                continue
            trial = current_set | {s}
            obj, ch = evaluate_set(trial)
            if obj > best_new_obj:
                best_new_obj = obj
                best_add = s
                best_new_chosen = ch
        if best_add is not None:
            current_set.add(best_add)
            cur_obj = best_new_obj
            cur_chosen = best_new_chosen
            improved = True

    # Also try removal pass: maybe removing some sum increases obj (if a color's
    # blocks all got displaced anyway). After max_nonoverlap, unused sums in set
    # don't actually appear; but we already only count distinct sums in chosen.
    # So C is naturally min. Good.

    if cur_obj > best_obj:
        best_obj = cur_obj
        # assign colors: map each distinct sum to a color id 1..B
        sums_used = sorted(set(s for _,_,s in cur_chosen))
        s2c = {s: i+1 for i, s in enumerate(sums_used)}
        best_output = [(l, r, s2c[s]) for (l, r, s) in cur_chosen]

    # Also try: just the single best sum
    if sorted_sums:
        for s, bl in sorted_sums[:5]:
            k = len(bl)
            obj = alpha*k - beta*1
            if obj > best_obj:
                best_obj = obj
                best_output = [(l, r, 1) for (l, r) in bl]

    # Output
    out_lines = []
    out_lines.append(str(len(best_output)))
    for (l, r, c) in best_output:
        out_lines.append(f"{l} {r} {c}")
    sys.stdout.write("\n".join(out_lines) + ("\n" if out_lines else ""))

solve()