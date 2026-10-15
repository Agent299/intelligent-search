"""Sliding-window query counter (doc p.169, Fig 5.10: 5-minute window sliding every 10 s). Tier 0.

Counts live in fixed buckets of `slide_s` seconds; the window is the last `window_s / slide_s` buckets, so a
slide costs one dict per bucket rather than a timestamp per search. One process holds the window: Kafka keys
query_logs by session, so a single consumer sees every search for a query (scaling out would need re-keying by
query or shared Redis buckets).
"""

from collections import Counter, deque


class SlidingWindow:
    def __init__(self, window_s: int = 300, slide_s: int = 10):
        if window_s % slide_s:
            raise ValueError("window_s must be a multiple of slide_s")
        self.slide_s = slide_s
        self.n_buckets = window_s // slide_s
        self.buckets: deque[tuple[int, Counter]] = deque()  # (bucket index, counts), oldest first
        self.totals: Counter = Counter()

    def _bucket(self, ts: float) -> int:
        return int(ts // self.slide_s)

    def _evict(self, now: float) -> None:
        oldest_kept = self._bucket(now) - self.n_buckets + 1
        while self.buckets and self.buckets[0][0] < oldest_kept:
            _, counts = self.buckets.popleft()
            self.totals.subtract(counts)
        self.totals = +self.totals  # drop zero counts

    def add(self, query: str, ts: float) -> None:
        b = self._bucket(ts)
        if self.buckets and b < self.buckets[-1][0]:
            b = self.buckets[-1][0]  # late event: count it in the current bucket rather than reorder
        if not self.buckets or self.buckets[-1][0] != b:
            self.buckets.append((b, Counter()))
        self.buckets[-1][1][query] += 1
        self.totals[query] += 1

    def counts(self, now: float) -> Counter:
        self._evict(now)
        return self.totals

    def trending(self, now: float, threshold: int) -> list[tuple[str, int]]:
        """Queries at or above the threshold in the current window, most searched first."""
        return [(q, n) for q, n in self.counts(now).most_common() if n >= threshold]
