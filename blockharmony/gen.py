#!/usr/bin/env python3
"""
gen.py — Test generator for IIMOC-P2 (Chromatic Block Harmony).

Usage:
    python3 gen.py <master_seed> <case_id>

master_seed=42  → full-size cases  (n up to 50000)
master_seed=999 → small viz cases  (n up to 30)
"""

import sys
import random
import math


def gen(master_seed: int, case_id: int):
    rng = random.Random(master_seed * 10007 + case_id)

    is_small = (master_seed == 999)

    if is_small:
        n     = rng.randint(12, 30)
        B     = rng.randint(1, 4)
        alpha = rng.randint(1, 10)
        beta  = rng.randint(0, 6)
    else:
        n     = int(10 ** rng.uniform(1, math.log10(10000)))
        B     = rng.randint(1, 15)
        alpha = rng.randint(1, 10**9)
        beta  = rng.randint(0, 10**9 // 2)

    a = [rng.randint(0, 50) for _ in range(n)]

    print(n, B, alpha, beta)
    print(*a)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen.py <master_seed> <case_id>", file=sys.stderr)
        sys.exit(1)
    gen(int(sys.argv[1]), int(sys.argv[2]))
