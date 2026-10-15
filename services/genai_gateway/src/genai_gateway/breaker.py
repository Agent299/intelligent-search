"""Per-model circuit breaker (doc p.158, p.180): after repeated failures stop calling a model for a
cool-down period so requests go straight to the next model in the chain instead of waiting on timeouts."""

import time
from enum import StrEnum


class State(StrEnum):
    closed = "closed"        # healthy: calls go through
    open = "open"            # tripped: calls are skipped until the cool-down ends
    half_open = "half_open"  # cool-down over: one trial call decides closed vs open


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_after_s: float = 30.0, clock=time.monotonic):
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_in_flight = False

    @property
    def state(self) -> State:
        if self._opened_at is None:
            return State.closed
        if self._clock() - self._opened_at >= self.reset_after_s:
            return State.half_open
        return State.open

    def allow(self) -> bool:
        state = self.state
        if state is State.closed:
            return True
        if state is State.half_open and not self._trial_in_flight:
            self._trial_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._trial_in_flight = False

    def record_failure(self) -> None:
        self._trial_in_flight = False
        if self.state is State.half_open:
            self._opened_at = self._clock()  # trial failed: re-open for another cool-down
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
