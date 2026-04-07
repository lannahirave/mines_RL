"""
Sum tree data structure for O(log n) prioritized sampling.
"""

import numpy as np


class SumTree:
    """
    Binary sum tree stored in a flat array.

    Leaf nodes hold priorities. Internal nodes hold sums of children.
    Supports O(log n) update, sample, and max queries.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.max_priority = 1.0

    @property
    def total(self) -> float:
        return self.tree[1]

    def update(self, index: int, priority: float):
        """Updates priority at leaf index. O(log n)."""
        if priority > self.max_priority:
            self.max_priority = priority

        pos = index + self.capacity
        self.tree[pos] = priority

        pos >>= 1
        while pos >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos >>= 1

    def update_batch(self, indices: np.ndarray, priorities: np.ndarray):
        """Updates multiple priorities. O(k log n)."""
        for i, p in zip(indices, priorities):
            self.update(int(i), float(p))

    def sample(self, batch_size: int) -> np.ndarray:
        """
        Samples batch_size leaf indices proportional to priorities. O(k log n).

        Returns:
            Array of leaf indices.
        """
        indices = np.empty(batch_size, dtype=np.int64)
        total = self.total

        if total <= 0 or not np.isfinite(total):
            return np.random.randint(0, max(1, self.capacity), size=batch_size)

        segment = total / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            indices[i] = self._retrieve(s)

        return indices

    def _retrieve(self, value: float) -> int:
        """Traverses tree to find leaf for given cumulative value."""
        pos = 1
        while pos < self.capacity:
            left = 2 * pos
            if value <= self.tree[left]:
                pos = left
            else:
                value -= self.tree[left]
                pos = left + 1
        return pos - self.capacity

    def __getitem__(self, index: int) -> float:
        return self.tree[index + self.capacity]
