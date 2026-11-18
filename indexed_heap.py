"""
indexed_heap.py - Indexed binary min-heap (a priority queue that supports updates).

Why not plain heapq?
  heapq cannot change the priority of, or remove, an item that is already inside.
  Tickets get escalated, re-prioritised, assigned or closed while waiting, so we need that.
  Common workaround = "lazy deletion" (leave dead entries, skip them later) -> heap grows with garbage.
  Here a hash map  item -> array position  makes every operation exact and garbage-free:

      push / update / remove / pop : O(log n)
      peek / membership test       : O(1)
      build from n pairs           : O(n)    (Floyd's heapify)
      peek_n(k) - top k in order   : O(k log k), independent of heap size

Used in 3 places:
  1. Ticket dispatch queue   key = (priority, sla_due, id)   -> most urgent ticket first
  2. SLA deadline heap       key = (sla_due, id)             -> earliest deadline first (breach detection)
  3. Agent availability heap key = (current_load, id)        -> least-loaded free agent first
"""
from heapq import heappop, heappush
from typing import Any, Hashable, Iterable, List, Tuple


class IndexedMinHeap:
    __slots__ = ("_keys", "_items", "_pos")

    def __init__(self) -> None:
        self._keys: List[Any] = []
        self._items: List[Hashable] = []
        self._pos: dict = {}

    # ------------------------------------------------------------- inspection
    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, item: Hashable) -> bool:
        return item in self._pos

    def key_of(self, item: Hashable) -> Any:
        return self._keys[self._pos[item]]

    def peek(self) -> Tuple[Hashable, Any]:
        if not self._keys:
            raise IndexError("peek from empty heap")
        return self._items[0], self._keys[0]

    def peek_n(self, n: int) -> List[Tuple[Hashable, Any]]:
        """Return the n smallest (item, key) pairs in order WITHOUT modifying the heap.
        Walks the heap with a small frontier heap: O(n log n) regardless of heap size."""
        size = len(self._keys)
        if n <= 0 or size == 0:
            return []
        keys, items = self._keys, self._items
        frontier = [(keys[0], 0)]
        out: List[Tuple[Hashable, Any]] = []
        while frontier and len(out) < n:
            key, i = heappop(frontier)
            out.append((items[i], key))
            left = 2 * i + 1
            if left < size:
                heappush(frontier, (keys[left], left))
                if left + 1 < size:
                    heappush(frontier, (keys[left + 1], left + 1))
        return out

    # ------------------------------------------------------------- mutation
    def push(self, item: Hashable, key: Any) -> None:
        """Insert, or change the key if the item is already present."""
        if item in self._pos:
            self.update(item, key)
            return
        i = len(self._keys)
        self._keys.append(key)
        self._items.append(item)
        self._pos[item] = i
        self._sift_up(i)

    def update(self, item: Hashable, key: Any) -> None:
        i = self._pos[item]
        old = self._keys[i]
        self._keys[i] = key
        if key < old:
            self._sift_up(i)
        elif old < key:
            self._sift_down(i)

    def pop(self) -> Tuple[Hashable, Any]:
        if not self._keys:
            raise IndexError("pop from empty heap")
        item, key = self._items[0], self._keys[0]
        self._remove_at(0)
        return item, key

    def remove(self, item: Hashable) -> Any:
        i = self._pos[item]
        key = self._keys[i]
        self._remove_at(i)
        return key

    def discard(self, item: Hashable) -> bool:
        if item in self._pos:
            self.remove(item)
            return True
        return False

    def build(self, pairs: Iterable[Tuple[Hashable, Any]]) -> None:
        """Bulk-load (item, key) pairs into an EMPTY heap in O(n)."""
        if self._keys:
            raise ValueError("build requires an empty heap")
        for item, key in pairs:
            self._pos[item] = len(self._keys)
            self._keys.append(key)
            self._items.append(item)
        for i in range(len(self._keys) // 2 - 1, -1, -1):
            self._sift_down(i)

    # ------------------------------------------------------------- internals
    def _remove_at(self, i: int) -> None:
        keys, items, pos = self._keys, self._items, self._pos
        del pos[items[i]]
        last_key = keys.pop()
        last_item = items.pop()
        if i == len(keys):          # removed the last slot
            return
        keys[i] = last_key
        items[i] = last_item
        pos[last_item] = i
        self._sift_up(i)
        self._sift_down(pos[last_item])

    def _sift_up(self, i: int) -> None:
        keys, items, pos = self._keys, self._items, self._pos
        key, item = keys[i], items[i]
        while i > 0:
            parent = (i - 1) >> 1
            if not key < keys[parent]:
                break
            keys[i] = keys[parent]
            items[i] = items[parent]
            pos[items[i]] = i
            i = parent
        keys[i] = key
        items[i] = item
        pos[item] = i

    def _sift_down(self, i: int) -> None:
        keys, items, pos = self._keys, self._items, self._pos
        n = len(keys)
        key, item = keys[i], items[i]
        while True:
            child = 2 * i + 1
            if child >= n:
                break
            if child + 1 < n and keys[child + 1] < keys[child]:
                child += 1
            if not keys[child] < key:
                break
            keys[i] = keys[child]
            items[i] = items[child]
            pos[items[i]] = i
            i = child
        keys[i] = key
        items[i] = item
        pos[item] = i

    # ------------------------------------------------------------- self-check (tests)
    def check_invariants(self) -> None:
        n = len(self._keys)
        assert n == len(self._items) == len(self._pos)
        for i in range(n):
            assert self._pos[self._items[i]] == i
            if i:
                assert not self._keys[i] < self._keys[(i - 1) >> 1]
