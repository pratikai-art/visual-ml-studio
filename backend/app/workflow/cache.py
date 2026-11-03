"""Execution cache keyed by (node type, params, upstream keys). Changing a node's
config changes its key and the keys of every downstream node (automatic invalidation)."""
from __future__ import annotations
import hashlib
import json
import threading
from collections import OrderedDict


class ExecutionCache:
    def __init__(self, max_items: int = 64):
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.max_items = max_items

    @staticmethod
    def key(node_type: str, params: dict, upstream_keys: list[str]) -> str:
        raw = json.dumps([node_type, params, upstream_keys], sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, k):
        with self._lock:
            v = self._d.get(k)
            if v is not None:
                self._d.move_to_end(k)
            return v

    def put(self, k, v):
        with self._lock:
            self._d[k] = v
            self._d.move_to_end(k)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def clear(self):
        with self._lock:
            self._d.clear()

    def __len__(self):
        return len(self._d)
