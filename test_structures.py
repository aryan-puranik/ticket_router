"""Randomised correctness tests for the data structures.  Run:  pytest -q test_structures.py"""
import random

import pytest

from audit import make_entry, verify_chain, AuditEntry
from bplus_tree import BPlusTree
from indexed_heap import IndexedMinHeap


@pytest.mark.parametrize("order", [4, 5, 6, 7, 16, 64])
def test_btree_random_ops_match_dict(order):
    rnd = random.Random(order)
    t, ref = BPlusTree(order), {}
    for step in range(6000):
        k = rnd.randrange(0, 1500)
        if rnd.random() < 0.55:
            assert t.insert(k, step) == (k not in ref)
            ref[k] = step
        else:
            assert t.delete(k) == (k in ref)
            ref.pop(k, None)
        if step % 250 == 0:
            t.check_invariants()
    t.check_invariants()
    assert len(t) == len(ref)
    assert list(t.items()) == sorted(ref.items())
    for k in range(0, 1500, 7):
        assert t.get(k) == ref.get(k)
        assert (k in t) == (k in ref)


@pytest.mark.parametrize("order", [4, 5, 8, 64])
def test_btree_scans(order):
    rnd = random.Random(1)
    keys = rnd.sample(range(10000), 2500)
    t = BPlusTree(order)
    for k in keys:
        t.insert(k, k * 2)
    sk = sorted(keys)
    for _ in range(200):
        lo, hi = sorted((rnd.randrange(-50, 10050), rnd.randrange(-50, 10050)))
        exp = [(k, k * 2) for k in sk if lo <= k <= hi]
        assert list(t.scan(lo, hi)) == exp
        assert list(t.scan(lo, hi, reverse=True)) == exp[::-1]
    assert list(t.scan(reverse=True)) == [(k, k * 2) for k in sk[::-1]]
    assert list(t.scan(lo=sk[10])) [0][0] == sk[10]
    assert t.min_key() == sk[0] and t.max_key() == sk[-1]


@pytest.mark.parametrize("n", [1, 2, 3, 63, 64, 65, 500, 4097, 20000])
@pytest.mark.parametrize("order", [4, 5, 64])
def test_bulk_load_then_mutate(n, order):
    t = BPlusTree(order)
    t.bulk_load((i * 3, i) for i in range(n))
    t.check_invariants()
    assert len(t) == n
    assert [k for k, _ in t.items()] == [i * 3 for i in range(n)]
    rnd = random.Random(n)
    ref = {i * 3: i for i in range(n)}
    for step in range(min(3000, n * 2)):
        k = rnd.randrange(0, n * 3 + 10)
        if rnd.random() < 0.5:
            t.insert(k, -step); ref[k] = -step
        else:
            t.delete(k); ref.pop(k, None)
    t.check_invariants()
    assert list(t.items()) == sorted(ref.items())


def test_bulk_load_rejects_unsorted_and_non_empty():
    t = BPlusTree()
    with pytest.raises(ValueError):
        t.bulk_load([(2, 0), (1, 0)])
    t.insert(1, 1)
    with pytest.raises(ValueError):
        t.bulk_load([(5, 0)])


def test_btree_tuple_prefix_search():
    t = BPlusTree(8)
    rows = [("alice@x.com", 1), ("alice@y.com", 2), ("bob@x.com", 3), ("alina@z.com", 4), ("alice@x.com", 9)]
    for r in rows:
        t.insert(r, r[1])
    lo, hi = ("ali", -1), ("ali\U0010ffff", 2 ** 63)
    assert [k for k, _ in t.scan(lo, hi)] == sorted(r for r in set(rows) if r[0].startswith("ali"))
    lo, hi = ("alice@x.com", -1), ("alice@x.com", 2 ** 63)
    assert [v for _, v in t.scan(lo, hi)] == [1, 9]


def test_btree_empty_and_drain():
    t = BPlusTree(4)
    assert t.get(1) is None and list(t.scan()) == [] and t.min_key() is None
    for i in range(200):
        t.insert(i, i)
    for i in range(200):
        assert t.delete(i)
        if i % 20 == 0:
            t.check_invariants()
    assert len(t) == 0 and t.height == 1
    t.check_invariants()


# ---------------------------------------------------------------------------- heap
def test_heap_random_ops_match_reference():
    rnd = random.Random(7)
    h, ref = IndexedMinHeap(), {}
    for step in range(8000):
        r = rnd.random()
        if r < 0.40:
            item, key = rnd.randrange(600), (rnd.randrange(5), rnd.randrange(1000), step)
            h.push(item, key); ref[item] = key
        elif r < 0.60 and ref:
            item = rnd.choice(list(ref))
            key = (rnd.randrange(5), rnd.randrange(1000), step)
            h.update(item, key); ref[item] = key
        elif r < 0.80 and ref:
            item = rnd.choice(list(ref))
            assert h.remove(item) == ref.pop(item)
        elif ref:
            item, key = h.pop()
            assert key == min(ref.values()) and ref.pop(item) == key
        if step % 200 == 0:
            h.check_invariants()
            assert len(h) == len(ref)
    h.check_invariants()
    out = []
    while h:
        out.append(h.pop()[1])
    assert out == sorted(ref.values())


def test_heap_build_peek_n_discard():
    rnd = random.Random(3)
    pairs = [(i, (rnd.randrange(100), i)) for i in range(1000)]
    h = IndexedMinHeap()
    h.build(pairs)
    h.check_invariants()
    expected = sorted((k, i) for i, k in pairs)
    assert [(k, i) for i, k in h.peek_n(25)] == expected[:25]
    assert len(h) == 1000                       # peek_n must not mutate
    assert h.peek_n(5000) and len(h.peek_n(5000)) == 1000
    assert h.discard(5) and not h.discard(5) and 5 not in h
    with pytest.raises(IndexError):
        IndexedMinHeap().pop()


# ---------------------------------------------------------------------------- audit chain
def test_audit_chain_detects_tampering():
    entries, prev = [], None
    for i in range(6):
        e = make_entry(prev, 42, 1000 + i, "alice", "STATUS_CHANGED", {"n": i})
        entries.append(e)
        prev = (e.seq, e.hash)
    assert verify_chain(entries) == (True, None)
    bad = list(entries)
    bad[2] = AuditEntry(*(bad[2][:3] + ("mallory",) + bad[2][4:]))      # edit actor
    assert verify_chain(bad) == (False, 3)
    assert verify_chain(entries[:2] + entries[3:])[0] is False            # delete one
