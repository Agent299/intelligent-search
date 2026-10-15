"""Process saturation metrics that work on macOS as well as Linux.

prometheus_client's default process collector reads /proc, so on macOS (where the services run in development)
it exports nothing. This collector uses only the standard library: CPU seconds and peak resident memory.
"""

import resource
import sys
import time

from prometheus_client import REGISTRY
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

_STARTED = time.time()


class AppProcessCollector:
    def __init__(self, service: str):
        self.service = service

    def collect(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is bytes on macOS, kilobytes on Linux
        maxrss = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
        cpu = CounterMetricFamily("app_process_cpu_seconds", "User + system CPU time", labels=["service"])
        cpu.add_metric([self.service], usage.ru_utime + usage.ru_stime)
        yield cpu
        rss = GaugeMetricFamily("app_process_max_rss_bytes", "Peak resident memory", labels=["service"])
        rss.add_metric([self.service], maxrss)
        yield rss
        up = GaugeMetricFamily("app_process_start_time_seconds", "Process start (unix time)", labels=["service"])
        up.add_metric([self.service], _STARTED)
        yield up


_collector: AppProcessCollector | None = None


def register_process_metrics(service: str) -> None:
    """One collector per process (one service per process in production). Later calls are no-ops: tests import
    several app factories into one process, and the registry rejects duplicate metric names."""
    global _collector
    if _collector is None:
        _collector = AppProcessCollector(service)
        REGISTRY.register(_collector)
