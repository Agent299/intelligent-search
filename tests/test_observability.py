import json
import re
from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import genai_gateway.app  # noqa: F401  (registers the gateway's metrics)
import pipelines.nrt_trends.detector  # noqa: F401
import search_service.app  # noqa: F401
from common.metrics import register_process_metrics
from common.schemas import EventType, SearchType
from search_service.events import INTERACTIONS, EventSink

ROOT = Path(__file__).resolve().parents[1]
ALERTS = ROOT / "ops/prometheus/alerts.yml"
DASHBOARD = ROOT / "ops/grafana/dashboards/search.json"
OURS = re.compile(r"\b((?:search|gateway|nrt|embedding|app_process|http)_[a-z0-9_]+)")
SUFFIXES = {"counter": ["_total", "_created"], "histogram": ["_bucket", "_count", "_sum", "_created"],
            "gauge": [], "summary": ["_count", "_sum"], "untyped": []}


def exported_names() -> set[str]:
    register_process_metrics("test")
    # the instrumentator registers http_* when the middleware stack is built, i.e. on the first request
    TestClient(search_service.app.create_app(engine=object())).get("/healthz")
    names = set()
    for fam in REGISTRY.collect():
        names.add(fam.name)
        names.update(fam.name + s for s in SUFFIXES.get(fam.type, []))
    return names


def referenced(exprs: list[str]) -> set[str]:
    """Metric names in PromQL, ignoring quoted label values like job="search_service"."""
    return {m for e in exprs for m in OURS.findall(re.sub(r'"[^"]*"', '""', e))}


def alert_exprs() -> list[str]:
    groups = yaml.safe_load(ALERTS.read_text())["groups"]
    return [r["expr"] for g in groups for r in g["rules"]]


def dashboard_exprs() -> list[str]:
    return [t["expr"] for p in json.loads(DASHBOARD.read_text())["panels"] for t in p.get("targets", [])]


def test_every_metric_in_alerts_and_dashboard_is_exported_by_the_code():
    missing = (referenced(alert_exprs()) | referenced(dashboard_exprs())) - exported_names()
    assert not missing, f"alerts/dashboard reference metrics no service exports: {sorted(missing)}"


def test_every_recording_rule_used_by_the_dashboard_is_defined():
    groups = yaml.safe_load(ALERTS.read_text())["groups"]
    recorded = {r["record"] for g in groups for r in g["rules"] if "record" in r}
    used = {m for e in dashboard_exprs() + alert_exprs() for m in re.findall(r"\b[a-z_]+:[a-z0-9_:]+\b", e)}
    assert used <= recorded, sorted(used - recorded)


def test_committed_dashboard_matches_its_generator():
    import importlib.util

    spec = importlib.util.spec_from_file_location("dashboard", ROOT / "ops/grafana/dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert json.loads(DASHBOARD.read_text()) == json.loads(json.dumps(mod.build())), \
        "run: uv run python ops/grafana/dashboard.py"


def test_alerts_have_severity_and_summary():
    for g in yaml.safe_load(ALERTS.read_text())["groups"]:
        for r in g["rules"]:
            if "alert" in r:
                assert r["labels"]["severity"] in {"critical", "warning"}, r["alert"]
                assert r["annotations"]["summary"], r["alert"]


class _Redis:
    def __init__(self):
        self.kv = {}

    async def set(self, key, value, ex=None):
        self.kv[key] = value

    async def get(self, key):
        return self.kv.get(key)


def _count(type_: str, surface: str) -> float:
    return INTERACTIONS.labels(type_, surface)._value.get()


async def test_interactions_are_counted_by_the_surface_that_showed_the_product():
    sink = EventSink(None, _Redis())
    await sink.search(search_id="s1", user_id=None, session_id=None, raw_query="laptop", filters={},
                      search_type=SearchType.hybrid, displayed=["p1"], device_type=None, pairs=["m1"])
    before = {k: _count(*k) for k in [("click", "results"), ("click", "pairs"), ("purchase", "other")]}
    await sink.interaction("s1", EventType.click, "p1")
    await sink.interaction("s1", EventType.click, "m1")
    await sink.interaction("s1", EventType.purchase, "zzz")
    assert _count("click", "results") == before[("click", "results")] + 1
    assert _count("click", "pairs") == before[("click", "pairs")] + 1
    assert _count("purchase", "other") == before[("purchase", "other")] + 1


def test_process_metrics_report_cpu_and_memory():
    register_process_metrics("test")
    cpu = REGISTRY.get_sample_value("app_process_cpu_seconds_total", {"service": "search_service"}) \
        or next(s.value for f in REGISTRY.collect() if f.name == "app_process_cpu_seconds" for s in f.samples
                if s.name.endswith("_total"))
    rss = next(s.value for f in REGISTRY.collect() if f.name == "app_process_max_rss_bytes" for s in f.samples)
    assert cpu > 0 and rss > 10_000_000
