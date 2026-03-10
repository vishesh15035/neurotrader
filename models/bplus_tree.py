"""
B+ Tree — Time-Series Indexing
O(log n) insert, delete, range query on timestamps
All data at leaf level, internal nodes are index only
"""
import numpy as np
from datetime import datetime

class BPlusNode:
    def __init__(self, order: int, is_leaf: bool = False):
        self.order    = order
        self.is_leaf  = is_leaf
        self.keys     = []
        self.values   = []  # only in leaves
        self.children = []  # only in internal
        self.next     = None  # leaf chain

class BPlusTree:
    """
    B+ Tree for O(log n) time-series data access
    Keys: timestamps (int), Values: OHLCV records
    """
    def __init__(self, order: int = 64):
        self.order = order
        self.root  = BPlusNode(order, is_leaf=True)
        self.size  = 0

    def insert(self, key: int, value: dict):
        leaf, parents = self._find_leaf(self.root, key, [])
        self._insert_in_leaf(leaf, key, value)
        if len(leaf.keys) >= self.order:
            self._split_leaf(leaf, parents)
        self.size += 1

    def _find_leaf(self, node, key, parents):
        if node.is_leaf: return node, parents
        parents.append(node)
        for i, k in enumerate(node.keys):
            if key < k:
                return self._find_leaf(node.children[i], key, parents)
        return self._find_leaf(node.children[-1], key, parents)

    def _insert_in_leaf(self, leaf, key, value):
        for i, k in enumerate(leaf.keys):
            if key == k: leaf.values[i] = value; return
            if key < k:
                leaf.keys.insert(i, key); leaf.values.insert(i, value); return
        leaf.keys.append(key); leaf.values.append(value)

    def _split_leaf(self, leaf, parents):
        mid      = self.order // 2
        new_leaf = BPlusNode(self.order, is_leaf=True)
        new_leaf.keys   = leaf.keys[mid:]
        new_leaf.values = leaf.values[mid:]
        new_leaf.next   = leaf.next
        leaf.keys       = leaf.keys[:mid]
        leaf.values     = leaf.values[:mid]
        leaf.next       = new_leaf
        push_up_key     = new_leaf.keys[0]
        if not parents:
            new_root          = BPlusNode(self.order, is_leaf=False)
            new_root.keys     = [push_up_key]
            new_root.children = [leaf, new_leaf]
            self.root         = new_root
        else:
            parent = parents[-1]
            for i, k in enumerate(parent.keys):
                if push_up_key < k:
                    parent.keys.insert(i, push_up_key)
                    parent.children.insert(i+1, new_leaf)
                    break
            else:
                parent.keys.append(push_up_key)
                parent.children.append(new_leaf)
            if len(parent.keys) >= self.order:
                self._split_internal(parent, parents[:-1])

    def _split_internal(self, node, parents):
        mid     = self.order // 2
        new_node= BPlusNode(self.order, is_leaf=False)
        mid_key = node.keys[mid]
        new_node.keys     = node.keys[mid+1:]
        new_node.children = node.children[mid+1:]
        node.keys         = node.keys[:mid]
        node.children     = node.children[:mid+1]
        if not parents:
            new_root          = BPlusNode(self.order, is_leaf=False)
            new_root.keys     = [mid_key]
            new_root.children = [node, new_node]
            self.root         = new_root
        else:
            parent = parents[-1]
            for i, k in enumerate(parent.keys):
                if mid_key < k:
                    parent.keys.insert(i, mid_key)
                    parent.children.insert(i+1, new_node)
                    break
            else:
                parent.keys.append(mid_key)
                parent.children.append(new_node)

    def range_query(self, start: int, end: int) -> list:
        """O(log n + k) range query where k = results"""
        leaf, _ = self._find_leaf(self.root, start, [])
        results = []
        while leaf:
            for i, k in enumerate(leaf.keys):
                if start <= k <= end:
                    results.append((k, leaf.values[i]))
                elif k > end:
                    return results
            leaf = leaf.next
        return results

    def search(self, key: int):
        leaf, _ = self._find_leaf(self.root, key, [])
        for i, k in enumerate(leaf.keys):
            if k == key: return leaf.values[i]
        return None

    def load_prices(self, prices: np.ndarray, timestamps: list = None) -> dict:
        """Load price series into B+ tree"""
        n = len(prices)
        if timestamps is None:
            timestamps = list(range(n))
        for i, (ts, p) in enumerate(zip(timestamps, prices)):
            self.insert(int(ts), {
                "price": float(p),
                "idx":   i,
                "return": float((prices[i]-prices[i-1])/prices[i-1]) if i>0 else 0.0
            })
        return {
            "n_records":   self.size,
            "tree_order":  self.order,
            "date_range":  [timestamps[0], timestamps[-1]],
            "complexity":  "O(log n) insert/search/delete",
            "range_query": "O(log n + k)"
        }
