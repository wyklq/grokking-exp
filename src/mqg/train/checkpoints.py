"""Log-uniform checkpoint scheduler.

Yields the next step to log/checkpoint based on a base list. We support
extending the list dynamically (for the adaptive T protocol).
"""
from __future__ import annotations

DEFAULT_LOG_STEPS: tuple[int, ...] = (
    100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
    100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000,
)


def steps_up_to(t_max: int, base=DEFAULT_LOG_STEPS) -> list[int]:
    """All log-uniform steps <= t_max."""
    return [s for s in base if s <= t_max]


class LogStepIterator:
    def __init__(self, base=DEFAULT_LOG_STEPS) -> None:
        self.steps = sorted(set(base))
        self._idx = 0

    def peek_next(self) -> int | None:
        return self.steps[self._idx] if self._idx < len(self.steps) else None

    def advance(self) -> None:
        self._idx += 1

    def reached(self, step: int) -> bool:
        nxt = self.peek_next()
        if nxt is not None and step >= nxt:
            self.advance()
            return True
        return False
