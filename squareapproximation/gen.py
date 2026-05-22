#!/usr/bin/env python3
"""
gen.py — Test generator for IIMOC-P4 (Rectangle Approximation).

Usage:
    python3 gen.py <master_seed> <case_id>

master_seed=42  → Organic emergent clumping cases (N in [1, 1000], coords in [-1000, 1000])
master_seed=999 → Small viz cases (N in [1, 20], coords in [1, 50])
"""

import sys
import random
import math


def count_connected_components(rects):
    """
    Uses a Union-Find data structure to determine how many distinct 
    overlapping clusters (connected components) emerged organically.
    """
    N = len(rects)
    parent = list(range(N))
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(N):
        for j in range(i + 1, N):
            A = rects[i]
            B = rects[j]
            # Check if interiors intersect
            if A[0] < B[2] and A[2] > B[0] and A[1] < B[3] and A[3] > B[1]:
                union(i, j)

    # Return the number of unique roots
    return len(set(find(i) for i in range(N)))


def generate_organic_rects(rng, N, bounds):
    """
    Generates N rectangles using a scale-invariant random-walk/mutation process.
    """
    min_c, max_c = bounds
    grid_span = max_c - min_c
    rects = []
    
    # Scale sizes down as N increases to prevent the grid from becoming one giant blob.
    # At N=1000, density_scale is ~11.2, shrinking base rectangles significantly
    # so they form textured islands instead of continental plates.
    density_scale = max(1.0, math.pow(N, 0.35))

    for i in range(N):
        # 30% chance to spawn completely randomly (seeds a new potential clump)
        if i == 0 or rng.random() < 0.30:
            x1 = rng.randint(min_c, max_c - 1)
            y1 = rng.randint(min_c, max_c - 1)
            
            # Dynamically scale initial sizes so they look proportional on any grid size,
            # while applying the density scale to maintain gaps at high N.
            min_w_add = max(1, int((grid_span / 200) / density_scale))
            max_w_add = max(2, int((grid_span / 5) / density_scale))
            
            x2 = min(max_c, x1 + rng.randint(min_w_add, max_w_add))
            y2 = min(max_c, y1 + rng.randint(min_w_add, max_w_add))
            rects.append((x1, y1, x2, y2))
            
        else:
            # 70% chance to mutate an existing rectangle
            base_rect = rng.choice(rects)
            bx1, by1, bx2, by2 = base_rect
            bw = bx2 - bx1
            bh = by2 - by1

            # Shift the center using Gaussian noise based on its own size
            cx = (bx1 + bx2) / 2.0 + rng.gauss(0, bw * 0.8)
            cy = (by1 + by2) / 2.0 + rng.gauss(0, bh * 0.8)

            # Scale the size using log-normal distribution (can shrink or grow)
            nw = max(1, int(bw * rng.lognormvariate(0, 0.4)))
            nh = max(1, int(bh * rng.lognormvariate(0, 0.4)))

            nx1 = int(cx - nw / 2.0)
            ny1 = int(cy - nh / 2.0)
            nx2 = nx1 + nw
            ny2 = ny1 + nh

            # Clamp back to the global coordinate bounds
            nx1 = max(min_c, min(max_c - 1, nx1))
            ny1 = max(min_c, min(max_c - 1, ny1))
            nx2 = max(nx1 + 1, min(max_c, nx2))
            ny2 = max(ny1 + 1, min(max_c, ny2))

            # Failsafe for collapsed rectangles
            if nx1 >= nx2 or ny1 >= ny2:
                nx2 = min(max_c, nx1 + 1)
                ny2 = min(max_c, ny1 + 1)

            rects.append((nx1, ny1, nx2, ny2))

    # Shuffle so the solver can't infer the growth order
    rng.shuffle(rects)
    return rects


def gen(master_seed: int, case_id: int):
    rng = random.Random(master_seed * 10007 + case_id)
    
    # Branching logic for the master_seed
    if master_seed == 999:
        N = rng.randint(1, 20)
        MIN_COORD = 1
        MAX_COORD = 50
    else:
        N = rng.randint(1, 1000)
        MIN_COORD = -1000
        MAX_COORD = 1000
        
    bounds = (MIN_COORD, MAX_COORD)
    
    # Generate the organically clumped rectangles
    rects = generate_organic_rects(rng, N, bounds)
    
    # Calculate emergent topology
    num_components = count_connected_components(rects)
    
    # Determine K based on the *actual* structural complexity
    base_k_min = num_components
    base_k_max = min(N, num_components + max(1, N // 5))
    
    K = rng.randint(base_k_min, base_k_max)
    K = max(1, min(N, K))
    
    print(f"{N} {K}")
    for r in rects:
        print(f"{r[0]} {r[1]} {r[2]} {r[3]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen.py <master_seed> <case_id>", file=sys.stderr)
        sys.exit(1)
    gen(int(sys.argv[1]), int(sys.argv[2]))