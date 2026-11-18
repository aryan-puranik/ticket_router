"""
benchmark.py - proves the data-structure choices with numbers (no database needed).

    python benchmark.py            # N = 200,000
    python benchmark.py 500000
"""
import bisect
import heapq
import random
import sys
import time

from bplus_tree import BPlusTree
from indexed_heap import IndexedMinHeap

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
rnd = random.Random(1)


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return time.perf_counter() - t0, r


def row(name, secs, ops, note=""):
    per = secs / ops * 1e6
    print(f"  {name:<52} {secs * 1000:10.1f} ms   {per:10.2f} us/op  {note}")


print(f"\n=== N = {N:,} tickets ===")

# ------------------------------------------------------------------ 1. building the id index
keys = list(range(1, N + 1))
shuffled = keys[:]
rnd.shuffle(shuffled)
print("\n[1] BUILD an ordered index")
tree = BPlusTree(64)
s, _ = timed(lambda: [tree.insert(k, k) for k in shuffled])
row("B+ tree: N random inserts", s, N)
tree2 = BPlusTree(64)
s, _ = timed(lambda: tree2.bulk_load((k, k) for k in keys))
row("B+ tree: bulk_load (sorted input)", s, N, "<- used at server start-up, O(n)")
sorted_list: list = []
s, _ = timed(lambda: [bisect.insort(sorted_list, k) for k in shuffled])
row("sorted list + bisect.insort: N random inserts", s, N, "<- O(n) memmove per insert, degrades as N grows")

# ------------------------------------------------------------------ 2. point lookup
print("\n[2] POINT LOOKUP by ticket id")
probe = [rnd.randrange(1, N + 1) for _ in range(50_000)]
s, _ = timed(lambda: [tree2.get(k) for k in probe])
row("B+ tree get()", s, len(probe))
flat = keys
few = probe[:200]
s, _ = timed(lambda: [flat.index(k) for k in few])
row("linear scan list.index() (200 lookups)", s, len(few), "<- O(n)")

# ------------------------------------------------------------------ 3. range query
print("\n[3] RANGE QUERY: 100 tickets between two ids / timestamps")
los = [rnd.randrange(1, N - 200) for _ in range(2000)]
s, _ = timed(lambda: [list(zip(range(100), tree2.scan(lo, lo + 99))) for lo in los])
row("B+ tree scan(lo, hi)  (2,000 queries)", s, len(los), "<- O(log n + k)")
s, _ = timed(lambda: [[k for k in flat if lo <= k <= lo + 99] for lo in los[:30]])
row("full filter of every ticket (30 queries)", s, 30, "<- O(n)")

# ------------------------------------------------------------------ 4. priority queue
print("\n[4] PRIORITY QUEUE under live changes (escalate / close / assign)")
pairs = [(i, (rnd.randrange(1, 5), rnd.randrange(10 ** 9), i)) for i in range(N)]
h = IndexedMinHeap()
s, _ = timed(lambda: h.build(pairs))
row("IndexedMinHeap.build (heapify)", s, N, "<- O(n)")
ids = [rnd.randrange(N) for _ in range(50_000)]
s, _ = timed(lambda: [h.update(i, (1, rnd.randrange(10 ** 9), i)) for i in ids])
row("IndexedMinHeap.update key (50,000)", s, len(ids), "<- O(log n)")
rem = rnd.sample(range(N), 50_000)
s, _ = timed(lambda: [h.remove(i) for i in rem])
row("IndexedMinHeap.remove arbitrary (50,000)", s, len(rem), "<- O(log n)")
s, _ = timed(lambda: [h.pop() for _ in range(50_000)])
row("IndexedMinHeap.pop (50,000)", s, 50_000, "<- O(log n)")
# naive: plain list, min() to find the most urgent, list.remove to delete
naive = [p[1] + (p[0],) for p in pairs[:20_000]]
s, _ = timed(lambda: [naive.remove(min(naive)) for _ in range(100)])
row("naive list: min()+remove() (100 ops, only 20k items)", s, 100, "<- O(n) each")
# lazy-deletion heapq: how much garbage does it accumulate when tickets are re-prioritised?
lazy = [(p[1], p[0]) for p in pairs]
heapq.heapify(lazy)
live = {i: k for i, k in pairs}
for i in ids:
    newk = (1, rnd.randrange(10 ** 9), i)
    live[i] = newk
    heapq.heappush(lazy, (newk, i))
print(f"  heapq + lazy deletion after {len(ids):,} re-prioritisations: {len(lazy):,} entries "
      f"for {len(live):,} live tickets ({len(lazy) - len(live):,} stale = garbage)")
print("  IndexedMinHeap keeps exactly one entry per live ticket - no garbage, no periodic rebuild")

# ------------------------------------------------------------------ 5. top-k peek
print("\n[5] 'show me the next 20 tickets to be worked' on a heap of N")
h2 = IndexedMinHeap()
h2.build(pairs)
s, _ = timed(lambda: [h2.peek_n(20) for _ in range(2000)])
row("IndexedMinHeap.peek_n(20)  (2,000 calls)", s, 2000, "<- O(k log k), N-independent")
s, _ = timed(lambda: [sorted(p[1] for p in pairs)[:20] for _ in range(2)])
row("sort everything then slice (2 calls)", s, 2, "<- O(n log n)")
print()
