# IIMOC-2026

My solutions to two optimization problems from IIMOC 2026. Both are heuristic
problems, so there is no "correct" answer, you just get scored on how good your
output is and you keep tweaking until the number stops going up.

Final standings:

| Problem | Score | Rank |
| --- | --- | --- |
| Rectangle Approximation | 99.19% | 14 / 280 |
| Chromatic Block Harmony | 99.92% | 14 / 275 |

## Files

```
square_approximation.py    solution to Rectangle Approximation
block_harmony.py           solution to Chromatic Block Harmony

squareapproximation/       problem package for Rectangle Approximation
  statement.txt              full problem statement
  gen.py                     official test generator (seed 42 = judge cases)
  config.yaml                limits, 5s and 512MB
  vis.py                     draws a picture of an input/output pair
  smalltest/                 sample cases, .in / .ans / .output / .png

blockharmony/              same layout, 3s and 512MB
```

Both solutions read stdin and write stdout, no arguments:

```
python3 square_approximation.py < squareapproximation/smalltest/1.in
python3 block_harmony.py < blockharmony/smalltest/1.in
```

To reproduce a judge case, `python3 gen.py 42 <case_id>` inside the problem
folder. Case ids run 1 to 30.

## Rectangle Approximation

You get N axis-aligned rectangles and a number K, and you have to output exactly
K rectangles whose union is as close as possible to the union of the input. Score
per case is `max(0, 1 - |A xor B| / |A|)` where A is the input union and B is
yours. N goes up to 1000, coordinates are integers in [-1000, 1000], and the
input rectangles are allowed to overlap each other.

### How my solution works

**Clustering.** Rectangles that don't touch can't help each other, so a union
find pass splits the input into connected components and each one gets solved
separately. This is the whole reason the thing is fast: the judge cases from
gen.py have lots of small clumps rather than one giant blob.

**Coordinate compression.** Only the x and y values that actually appear in the
input ever matter, so every grid in the file is a compressed grid where one cell
stands for a whole rectangle of real area. Cell areas get carried around as
`cell_dx` and `cell_dy` instead of being assumed to be 1.

**Building an answer (`build_up`).** Greedy, one rectangle at a time, always
taking whichever rectangle helps the most. The "helps the most" part has two
possible definitions and I tried both:

- Max gain, where a rectangle's value is (new area of A it covers) minus (area
  outside A it wastes). If you write +area on cells inside A and -area on cells
  outside, this is exactly the max-sum subrectangle, which 2D Kadane solves in
  O(nx^2 * ny). That's `max_sum_subrect`.
- Zero excess, where you only consider rectangles that lie entirely inside A, so
  the value is just plain area. That's a largest-rectangle-in-a-histogram sweep
  in O(nx * ny), which is `largest_rect_in_A`.

Zero excess won, and not by a little. It was about 25 times faster (8 to 10 ms
per call versus 200 to 250 ms) and it also scored better. Letting a rectangle
swallow some excess looks good locally but it messes up everything you place
afterward. `KADANE_THRESH` is set to 0 for that reason, so Kadane only runs on
tiny inputs where the global path is used.

**Splitting the budget across clusters.** Every rectangle `build_up` places
comes with a gain, and gains inside a cluster are decreasing, so allocating K
across clusters is just a heap: push each cluster's next gain, pop K times.

The important detail is that every cluster starts at 0 budget instead of getting
one rectangle for free. That sounds backwards, since a cluster with 0 rectangles
contributes its entire area to the error, but giving everyone a floor of 1 is
what starves the big clusters when K is close to the number of clusters. Letting
a big cluster's second and third rectangle beat a tiny cluster's first was worth
+0.0245 average, which is one of the biggest single wins in the file.

**Drop-down (`drop_down_cluster`).** The opposite approach: start with all the
input rectangles of the cluster and repeatedly delete the one whose removal costs
the least, until only k are left. This beats build-up when K/N is high and the
input rectangles are already good approximations of themselves. Both get run and
whichever scores better gets kept.

**Edge sliding (`optimize_edges`).** This was the single biggest improvement,
about +0.011 average on its own. For each output rectangle, take one edge at a
time, hold the other three fixed, and scan every position that edge could be at.
Moving out swallows nearby uncovered area but drags in any excess it crosses.
Moving in trims excess off the end. Take the best position, and only if it
strictly improves. The bookkeeping is per cell, with `signed` being +area inside
A and -area outside, and `coverage` counting how many output rectangles sit on
the cell:

- covering a cell nobody else covers changes the score by -signed
- uncovering a cell only this rectangle covers changes it by +signed
- cells another rectangle also covers don't change anything either way

Since only improving moves get applied, running this more can never hurt.

**Relocation (`refine_cluster`).** Find the output rectangle covering the least
area that nothing else covers, delete it, and check whether a bigger rectangle
now fits in the freed space. Keep the swap only if the new one is strictly worth
more. This is what fixes the case where the greedy build-up boxed itself in.

Edge sliding and relocation alternate as coordinate descent until the cluster's
time slice runs out. Each is monotone in score so more rounds only help, and the
time budget is handed out to clusters in proportion to their rectangle budget.

**Special cases.** If K >= N you can just print the input and score exactly 1.0,
which is worth checking first. If the whole input is small enough that all
coordinate pairs fit in memory, it gets solved as one global instance instead of
clustering.

**Safety caps.** `CELL_CAP` and `WORK_CAP` bail out to a cheap answer if a
cluster's compressed grid is enormous. This never fires on real judge data, since
gen.py shrinks the rectangles as N grows, but a pathological dense cluster would
otherwise blow the time limit.

### Score progression

Local benchmark over the judge generator, seed 42, cases 1 to 30:

| Change | Score |
| --- | --- |
| baseline greedy | 0.9326 |
| fixed the `build_up` gains bug | 0.9119 to 0.9267 |
| from-scratch budget allocation | 0.9458 |
| edge sliding both directions | 0.9693 |

That last number is a 3.94% improvement over the baseline, which does not sound
like much until you look at the leaderboard and realize that's most of the gap
between rank 60 and rank 14.

### Things that didn't work

**Mixing Kadane and histogram.** I spent a while assuming the smarter primitive
(max gain, allowed to take beneficial excess) had to be better and only used the
histogram version as a speed fallback for big grids. It's just worse. Set to 0
and never looked back.

**merge_to_budget.** A pass that merges pairs of output rectangles when their
bounding box is cheaper than keeping both. It genuinely improved individual
instances, which is why I kept it for a while, but it cost enough runtime that
the full pipeline average went down, 0.9693 to 0.9691. Fully removed. Lesson:
measure the pipeline, not the function.

**Raising AUTO_THRESH to 3M.** Let bigger clusters use the expensive path. Caused
TLE-induced score drops on several seeds. Reverted.

**A gains diagnostic that lied to me.** I had a probe reporting that a greedy
pass found 72 rectangles when the real count was 165. The time guard was firing
during the probe itself, so the probe was measuring its own interruption. Worth
remembering that instrumentation can be the bug.

**The TLE.** At one point a single case went from 0.07s to 5.50s. The cause was
the per-cluster time budget being computed too loosely, so a handful of clusters
ate everything and the rest ran with no guard at all. The fix is the
`remaining_b` proportional split in `main()`.

## Chromatic Block Harmony

You get an array of n integers and up to B colors. You pick non-overlapping
contiguous blocks and color them, with the rule that all blocks sharing a color
must have the same sum. Objective is `alpha * k - beta * C` where k is the number
of blocks and C is the number of distinct colors used. Choosing a color costs you
beta, so a color only pays for itself if enough blocks use it. n goes up to 50000,
B up to 20, and array values are in [0, 50].

### How my solution works

The key realization is that a color is really just "a sum value", so the whole
problem is: pick at most B sums, then pack as many non-overlapping blocks with
those sums as you can. Once the sums are chosen, packing is just interval
scheduling, which greedy solves exactly by taking blocks in order of right
endpoint.

**Finding blocks with a given sum.** Prefix sums plus binary search. For each
left endpoint L, the right endpoint is wherever `P[L-1] + s` shows up in P, if it
does. Results are cached per sum, since the same sums get queried constantly.

**Picking candidate sums.** Every value that appears in the array is a candidate,
because a single element is a valid block. On top of that, sums of windows of
length 2 to 20 that appear often enough get added. Then all candidates get ranked
by `greedy_count`, an O(n) estimate of how many disjoint blocks that sum alone
can fit, and only the top ones survive.

**Phase 1, single elements only.** If you only use single-element blocks, they
can never overlap, so the objective is exactly `alpha * sum(freq) - beta * C` and
you can pick greedily by frequency with no scheduling needed. This gives a solid
baseline instantly and is often already optimal when beta is large.

**Phase 2, greedy forward pass.** Add colors one at a time, each time taking the
one that improves the objective most. The trick that makes this affordable is
keeping the merged block list incrementally, so testing a candidate is a linear
merge rather than rebuilding everything.

**Phase 3, swap refinement.** Try replacing each chosen sum with an unchosen one.
Prefix and suffix merges are precomputed so removing color i from the set is O(1)
lookups plus one merge rather than re-merging all B lists.

### Things that didn't work

**Unbounded swap refinement.** Phase 3 originally ran until it converged. On
n=10000 it found zero improvements and cost an extra 0.72 seconds, and on judge
case 33 it TLEd outright. It's now guarded by a 2.4 second budget against a 3
second limit, plus a size check, and it bails the moment it stops finding swaps.

**The tagged_select bottleneck.** Profiling showed 2.1 seconds across 1590 calls
in the selection routine alone. Caching the block lists per sum and merging
incrementally instead of re-sorting the full tagged list each time is what made
the rest of the search affordable.
