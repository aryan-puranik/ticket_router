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

    
    def delete(self, key: Any) -> bool:
        """Remove key. Returns True if it existed."""
        path: List[Tuple[_Node, int]] = []
        node = self._root
        for _ in range(self._height - 1):
            i = bisect_right(node.keys, key)
            path.append((node, i))
            node = node.children[i]
        leaf: _Leaf = node
        keys = leaf.keys
        i = bisect_left(keys, key)
        if i >= len(keys) or keys[i] != key:
            return False
        del keys[i]
        del leaf.vals[i]
        self._size -= 1
        if path and len(keys) < self._min:
            self._fix_leaf(leaf, path)
        return True

    def _fix_leaf(self, leaf: _Leaf, path: List[Tuple[_Node, int]]) -> None:
        parent, idx = path.pop()
        left = parent.children[idx - 1] if idx > 0 else None
        right = parent.children[idx + 1] if idx + 1 < len(parent.children) else None

        if left is not None and len(left.keys) > self._min:          # borrow from left
            leaf.keys.insert(0, left.keys.pop())
            leaf.vals.insert(0, left.vals.pop())
            parent.keys[idx - 1] = leaf.keys[0]
            return
        if right is not None and len(right.keys) > self._min:        # borrow from right
            leaf.keys.append(right.keys.pop(0))
            leaf.vals.append(right.vals.pop(0))
            parent.keys[idx] = right.keys[0]
            return
        if left is not None:                                         # merge into left
            left.keys.extend(leaf.keys)
            left.vals.extend(leaf.vals)
            left.next = leaf.next
            if leaf.next is not None:
                leaf.next.prev = left
            del parent.keys[idx - 1]
            del parent.children[idx]
        else:                                                        # merge right into leaf
            leaf.keys.extend(right.keys)
            leaf.vals.extend(right.vals)
            leaf.next = right.next
            if right.next is not None:
                right.next.prev = leaf
            del parent.keys[idx]
            del parent.children[idx + 1]
        self._fix_internal(parent, path)

    def _fix_internal(self, node: _Node, path: List[Tuple[_Node, int]]) -> None:
        while True:
            if not path:                       # node is the root
                if not node.keys:              # root lost its last separator -> shrink tree
                    self._root = node.children[0]
                    self._height -= 1
                return
            if len(node.keys) >= self._min:
                return
            parent, idx = path.pop()
            left = parent.children[idx - 1] if idx > 0 else None
            right = parent.children[idx + 1] if idx + 1 < len(parent.children) else None

            if left is not None and len(left.keys) > self._min:      # rotate right
                node.keys.insert(0, parent.keys[idx - 1])
                node.children.insert(0, left.children.pop())
                parent.keys[idx - 1] = left.keys.pop()
                return
            if right is not None and len(right.keys) > self._min:    # rotate left
                node.keys.append(parent.keys[idx])
                node.children.append(right.children.pop(0))
                parent.keys[idx] = right.keys.pop(0)
                return
            if left is not None:                                     # merge node into left
                left.keys.append(parent.keys[idx - 1])
                left.keys.extend(node.keys)
                left.children.extend(node.children)
                del parent.keys[idx - 1]
                del parent.children[idx]
            else:                                                    # merge right into node
                node.keys.append(parent.keys[idx])
                node.keys.extend(right.keys)
                node.children.extend(right.children)
                del parent.keys[idx]
                del parent.children[idx + 1]
            node = parent

    # ------------------------------------------------------------------ bulk load
    def bulk_load(self, items: Iterable[Tuple[Any, Any]]) -> None:
        """Build the tree bottom-up from (key, value) pairs sorted by strictly increasing key.
        O(n) - far faster than n inserts, and nodes are packed evenly. Tree must be empty."""
        if self._size:
            raise ValueError("bulk_load requires an empty tree")
        items = list(items)
        n = len(items)
        if n == 0:
            return
        for i in range(n - 1):
            if not items[i][0] < items[i + 1][0]:
                raise ValueError("bulk_load input must be strictly increasing by key")

        n_leaves = -(-n // self._max)
        base, extra = divmod(n, n_leaves)
        level: List[Any] = []
        prev: Optional[_Leaf] = None
        pos = 0
        for j in range(n_leaves):
            size = base + (1 if j < extra else 0)
            chunk = items[pos:pos + size]
            pos += size
            leaf = _Leaf()
            leaf.keys = [k for k, _ in chunk]
            leaf.vals = [v for _, v in chunk]
            leaf.prev = prev
            if prev is not None:
                prev.next = leaf
            prev = leaf
            level.append(leaf)

        firsts = [lf.keys[0] for lf in level]   # smallest key under each node of `level`
        height = 1
        fanout = self._max + 1
        while len(level) > 1:
            c = len(level)
            n_par = -(-c // fanout)
            base, extra = divmod(c, n_par)
            parents: List[_Node] = []
            parent_firsts: List[Any] = []
            pos = 0
            for j in range(n_par):
                size = base + (1 if j < extra else 0)
                node = _Node()
                node.children = level[pos:pos + size]
                node.keys = firsts[pos + 1:pos + size]
                parent_firsts.append(firsts[pos])
                parents.append(node)
                pos += size
            level, firsts = parents, parent_firsts
            height += 1
        self._root = level[0]
        self._height = height
        self._size = n

    # ------------------------------------------------------------------ self-check (used by tests)
    def check_invariants(self) -> None:
        leaves: List[_Leaf] = []

        def walk(node: Any, depth: int, lo: Any, hi: Any, is_root: bool) -> int:
            ks = node.keys
            if depth == self._height:
                assert isinstance(node, _Leaf)
                assert len(ks) == len(node.vals)
                assert all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))
                if ks:
                    if lo is not None:
                        assert ks[0] >= lo
                    if hi is not None:
                        assert ks[-1] < hi
                assert len(ks) <= self._max
                if not is_root:
                    assert len(ks) >= self._min
                leaves.append(node)
                return len(ks)
            assert isinstance(node, _Node)
            assert len(node.children) == len(ks) + 1
            assert all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))
            assert len(ks) <= self._max
            assert (len(ks) >= 1) if is_root else (len(ks) >= self._min)
            total = 0
            for i, ch in enumerate(node.children):
                clo = lo if i == 0 else ks[i - 1]
                chi = hi if i == len(ks) else ks[i]
                total += walk(ch, depth + 1, clo, chi, False)
            return total

        total = walk(self._root, 1, None, None, True)
        assert total == self._size, (total, self._size)
        chain: List[_Leaf] = []
        leaf, prev = self._leftmost(), None
        while leaf is not None:
            assert leaf.prev is prev
            chain.append(leaf)
            prev, leaf = leaf, leaf.next
        assert len(chain) == len(leaves) and all(a is b for a, b in zip(chain, leaves))
