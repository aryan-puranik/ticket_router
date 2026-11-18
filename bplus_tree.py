"""
bplus_tree.py - In-memory B+ Tree.

Why a B+ tree for the ticket router?
  * All data lives in the LEAVES; leaves are chained in a doubly linked list.
  * Point lookup / insert / delete : O(log_B n)   (B = order, default 64 -> tree stays ~3-4 levels for millions of keys)
  * Range scan                     : O(log_B n + k) -> descend once, then walk the leaf chain
  * Prefix search on tuple keys    : a range scan, e.g. ("john", -inf) .. ("john\\uffff", +inf)
  * Bulk load (sorted input)       : O(n), builds a densely packed tree bottom-up (used at server start-up)

Node capacity (order = B):
  max keys per node = B - 1, min keys per non-root node = (B - 1) // 2
  Leaf split   -> "copy up" the first key of the new right leaf.
  Internal split -> "push up" the middle key.
  Delete       -> borrow from a sibling if possible, otherwise merge (cascades upward).
"""
from bisect import bisect_left, bisect_right
from typing import Any, Iterable, Iterator, List, Optional, Tuple


class _Leaf:
    __slots__ = ("keys", "vals", "next", "prev")

    def __init__(self) -> None:
        self.keys: List[Any] = []
        self.vals: List[Any] = []
        self.next: Optional["_Leaf"] = None
        self.prev: Optional["_Leaf"] = None


class _Node:
    __slots__ = ("keys", "children")

    def __init__(self) -> None:
        self.keys: List[Any] = []
        self.children: List[Any] = []


class BPlusTree:
    def __init__(self, order: int = 64) -> None:
        if order < 4:
            raise ValueError("order must be >= 4")
        self.order = order
        self._max = order - 1          # max keys in any node
        self._min = self._max // 2     # min keys in a non-root node
        self._root: Any = _Leaf()
        self._height = 1               # number of levels (a lone leaf = 1)
        self._size = 0

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: Any) -> bool:
        leaf = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        return i < len(leaf.keys) and leaf.keys[i] == key

    @property
    def height(self) -> int:
        return self._height

    def _find_leaf(self, key: Any) -> _Leaf:
        node = self._root
        for _ in range(self._height - 1):
            node = node.children[bisect_right(node.keys, key)]
        return node

    def _leftmost(self) -> _Leaf:
        node = self._root
        for _ in range(self._height - 1):
            node = node.children[0]
        return node

    def _rightmost(self) -> _Leaf:
        node = self._root
        for _ in range(self._height - 1):
            node = node.children[-1]
        return node

    # ------------------------------------------------------------------ search
    def get(self, key: Any, default: Any = None) -> Any:
        leaf = self._find_leaf(key)
        keys = leaf.keys
        i = bisect_left(keys, key)
        if i < len(keys) and keys[i] == key:
            return leaf.vals[i]
        return default

    def min_key(self) -> Any:
        leaf = self._leftmost()
        return leaf.keys[0] if leaf.keys else None

    def max_key(self) -> Any:
        leaf = self._rightmost()
        return leaf.keys[-1] if leaf.keys else None

    def scan(self, lo: Any = None, hi: Any = None, *, reverse: bool = False) -> Iterator[Tuple[Any, Any]]:
        """Yield (key, value) for lo <= key <= hi (None = unbounded).
        reverse=True yields in descending key order. Cost: O(log n + k)."""
        if not reverse:
            if lo is None:
                leaf, i = self._leftmost(), 0
            else:
                leaf = self._find_leaf(lo)
                i = bisect_left(leaf.keys, lo)
            while leaf is not None:
                keys, vals = leaf.keys, leaf.vals
                n = len(keys)
                while i < n:
                    k = keys[i]
                    if hi is not None and k > hi:
                        return
                    yield k, vals[i]
                    i += 1
                leaf, i = leaf.next, 0
        else:
            if hi is None:
                leaf = self._rightmost()
                i = len(leaf.keys) - 1
            else:
                leaf = self._find_leaf(hi)
                i = bisect_right(leaf.keys, hi) - 1
            while leaf is not None:
                keys, vals = leaf.keys, leaf.vals
                while i >= 0:
                    k = keys[i]
                    if lo is not None and k < lo:
                        return
                    yield k, vals[i]
                    i -= 1
                leaf = leaf.prev
                i = len(leaf.keys) - 1 if leaf is not None else -1

    def items(self) -> Iterator[Tuple[Any, Any]]:
        return self.scan()

    def keys(self) -> Iterator[Any]:
        for k, _ in self.scan():
            yield k

    # ------------------------------------------------------------------ insert
    def insert(self, key: Any, value: Any) -> bool:
        """Insert or overwrite. Returns True if the key is new."""
        path: List[Tuple[_Node, int]] = []
        node = self._root
        for _ in range(self._height - 1):
            i = bisect_right(node.keys, key)
            path.append((node, i))
            node = node.children[i]
        leaf: _Leaf = node
        keys = leaf.keys
        i = bisect_left(keys, key)
        if i < len(keys) and keys[i] == key:
            leaf.vals[i] = value
            return False
        keys.insert(i, key)
        leaf.vals.insert(i, value)
        self._size += 1
        if len(keys) <= self._max:
            return True

        # ---- split the overflowing leaf (copy-up)
        mid = len(keys) // 2
        right = _Leaf()
        right.keys = keys[mid:]
        right.vals = leaf.vals[mid:]
        del keys[mid:]
        del leaf.vals[mid:]
        right.next = leaf.next
        if leaf.next is not None:
            leaf.next.prev = right
        right.prev = leaf
        leaf.next = right
        sep: Any = right.keys[0]
        child: Any = right

        # ---- propagate splits upward (push-up)
        while path:
            parent, idx = path.pop()
            parent.keys.insert(idx, sep)
            parent.children.insert(idx + 1, child)
            if len(parent.keys) <= self._max:
                return True
            mid = len(parent.keys) // 2
            sep = parent.keys[mid]
            new = _Node()
            new.keys = parent.keys[mid + 1:]
            new.children = parent.children[mid + 1:]
            del parent.keys[mid:]
            del parent.children[mid + 1:]
            child = new

        root = _Node()
        root.keys = [sep]
        root.children = [self._root, child]
        self._root = root
        self._height += 1
        return True

