"""Generates ops/grafana/dashboards/search.json (Grafana provisions it on start).

    uv run python ops/grafana/dashboard.py

Design rules (validated palette, dark theme): one y-axis per panel; every serving path / cache tier keeps the
same color in every panel (color follows the entity, set by name override, never by series order); percentiles
are an ordered set, so they use one blue ramp light -> dark instead of category hues; status colors (good /
warning / critical) appear only on thresholds; 2 px lines; legends on every multi-series panel.
"""

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards" / "search.json"
DS = {"type": "prometheus", "uid": "prometheus"}

# Categorical slots, dark-mode steps (validated: ALL CHECKS PASS on #1a1a19, in this order)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                                                      "#008300", "#9085e9")
PATH_COLORS = {"results_cache": BLUE, "l1": ORANGE, "l2": AQUA, "warm": YELLOW, "fast_path": MAGENTA,
               "fallback": GREEN}
TIER_COLORS = {"results": BLUE, "l1": ORANGE, "l2": AQUA, "warm_exact": YELLOW, "warm_similar": VIOLET}
PERCENTILE_RAMP = {"p50": "#86b6ef", "p90": "#5598e7", "p95": "#2a78d6", "p99": "#1c5cab"}  # blue 250 -> 550
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"  # status palette, thresholds only

_id = 0


def _next_id() -> int:
    global _id
    _id += 1
    return _id


def target(expr: str, legend: str = "", ref: str = "A") -> dict:
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref, "range": True}


def overrides(colors: dict[str, str]) -> list[dict]:
    return [{"matcher": {"id": "byName", "options": name},
             "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}
            for name, color in colors.items()]


def thresholds(*steps: tuple[float | None, str]) -> dict:
    return {"mode": "absolute", "steps": [{"value": v, "color": c} for v, c in steps]}


def stat(title: str, expr: str, unit: str, x: int, y: int, w: int = 4, desc: str = "",
         steps: tuple = ((None, "text"),), decimals: int | None = None) -> dict:
    return {
        "id": _next_id(), "type": "stat", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": 4}, "targets": [target(expr)],
        "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals, "thresholds": thresholds(*steps),
                                     "color": {"mode": "thresholds"}}, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area",
                    "textMode": "value", "justifyMode": "center"},
    }


def series(title: str, targets: list[dict], unit: str, x: int, y: int, w: int = 12, h: int = 8, desc: str = "",
           colors: dict[str, str] | None = None, stack: bool = False, threshold_lines: tuple = (),
           min_: float | None = 0, max_: float | None = None) -> dict:
    steps = [(None, "transparent")] + list(threshold_lines)
    return {
        "id": _next_id(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
        "fieldConfig": {
            "defaults": {
                "unit": unit, "min": min_, "max": max_, "color": {"mode": "palette-classic"},
                "thresholds": thresholds(*steps),
                "custom": {"lineWidth": 2, "fillOpacity": 25 if stack else 0, "showPoints": "never",
                           "spanNulls": True, "axisSoftMin": 0, "gradientMode": "none",
                           "stacking": {"mode": "normal" if stack else "none"},
                           "thresholdsStyle": {"mode": "dashed" if threshold_lines else "off"}},
            },
            "overrides": overrides(colors or {}),
        },
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
    }


def row(title: str, y: int) -> dict:
    return {"id": _next_id(), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


def build() -> dict:
    p: list[dict] = []
    y = 0
    p.append(row("Business (doc p.180): is search working for shoppers?", y))
    y += 1
    p += [
        stat("Searches / s", "search:requests:rate5m", "reqps", 0, y, decimals=2),
        stat("Search CTR (1 h)", "search:ctr:1h", "percentunit", 4, y, decimals=1,
             desc="Clicks on result products / searches that showed results"),
        stat("Conversion (1 h)", "search:conversion:1h", "percentunit", 8, y, decimals=2,
             desc="Purchases attributed to a search / searches"),
        stat("Pairs-well-with CTR (1 h)", "search:discovery_ctr:1h", "percentunit", 12, y, decimals=1,
             desc="Clicks on 'pairs well with' products / searches that showed them"),
        stat("Zero-result rate", "search:zero_result_ratio:5m", "percentunit", 16, y, decimals=1,
             steps=((None, GOOD), (0.05, WARNING), (0.10, CRITICAL)), desc="Alert at 10% for 15 min"),
        stat("LLM fallback rate", "search:fallback_ratio:5m", "percentunit", 20, y, decimals=1,
             steps=((None, GOOD), (0.05, WARNING), (0.20, CRITICAL)),
             desc="Doc expects ~1% of traffic on the LLM path; alert at 20% for 15 min"),
    ]
    y += 4

    p.append(row("Latency (NFR: P90 < 1 s, P99 < 5 s)", y))
    y += 1
    p.append(series("Search latency percentiles", [
        target("search:latency_seconds:p50_5m", "p50", "A"), target("search:latency_seconds:p90_5m", "p90", "B"),
        target("search:latency_seconds:p95_5m", "p95", "C"), target("search:latency_seconds:p99_5m", "p99", "D")],
        "s", 0, y, colors=PERCENTILE_RAMP, threshold_lines=((1, WARNING), (5, CRITICAL)),
        desc="Dashed lines: the P90 (1 s) and P99 (5 s) targets"))
    p.append(series("P95 latency by serving path", [target(
        "histogram_quantile(0.95, sum by (le, served_by) (rate(search_latency_seconds_bucket[5m])))",
        "{{served_by}}")], "s", 12, y, colors=PATH_COLORS,
        desc="Cache tiers should sit in milliseconds; fallback carries the LLM call"))
    y += 8

    p.append(row("Traffic and caches", y))
    y += 1
    p.append(series("Searches / s by serving path", [target(
        "sum by (served_by) (rate(search_requests_total[5m]))", "{{served_by}}")],
        "reqps", 0, y, colors=PATH_COLORS, stack=True))
    p.append(series("Cache hit ratio by tier", [target(
        'sum by (tier) (rate(search_cache_lookups_total{result="hit"}[5m])) '
        "/ sum by (tier) (rate(search_cache_lookups_total[5m]))", "{{tier}}")],
        "percentunit", 12, y, colors=TIER_COLORS, max_=1,
        desc="results: hot queries only; warm_similar: only consulted below the confidence gate"))
    y += 8

    p.append(row("LLM gateway", y))
    y += 1
    p.append(series("LLM spend, $ / hour by task", [target("gateway:llm_cost_usd_per_hour:1h", "{{task}}")],
                    "currencyUSD", 0, y, w=8, stack=True, threshold_lines=((1, WARNING),),
                    desc="Alert above $1/h for 15 min"))
    p.append(series("LLM attempts / s by outcome", [target(
        "sum by (outcome) (rate(gateway_llm_attempts_total[5m]))", "{{outcome}}")],
        "reqps", 8, y, w=8, colors={"ok": BLUE, "invalid": ORANGE, "timeout": AQUA, "error": YELLOW,
                                    "refusal": MAGENTA, "truncated": GREEN, "skipped_open": VIOLET}))
    p.append(series("LLM P95 latency by task", [target(
        "histogram_quantile(0.95, sum by (le, task) (rate(gateway_llm_latency_seconds_bucket[5m])))", "{{task}}")],
        "s", 16, y, w=8))
    y += 8
    p.append({
        "id": _next_id(), "type": "state-timeline", "title": "Circuit breaker state by model", "datasource": DS,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": 5}, "targets": [target("gateway_breaker_state", "{{model}}")],
        "fieldConfig": {"defaults": {
            "mappings": [{"type": "value", "options": {
                "0": {"text": "closed", "color": GOOD}, "1": {"text": "half open", "color": WARNING},
                "2": {"text": "open", "color": CRITICAL}}}],
            "color": {"mode": "thresholds"}, "thresholds": thresholds((None, GOOD))}, "overrides": []},
        "options": {"showValue": "auto", "legend": {"showLegend": True, "displayMode": "list",
                                                    "placement": "bottom"}},
    })
    y += 5

    p.append(row("Ranking, discovery, trends", y))
    y += 1
    p.append(series("Re-rank P95 latency", [target(
        "histogram_quantile(0.95, sum by (le, ranker) (rate(search_rerank_seconds_bucket[5m])))", "{{ranker}}")],
        "s", 0, y, w=8, colors={"lightgbm": BLUE, "linear": ORANGE}))
    p.append(series("Pairs-well-with lookups / s", [target(
        "sum by (result) (rate(search_discovery_lookups_total[5m]))", "{{result}}")],
        "reqps", 8, y, w=8, colors={"hit": BLUE, "miss": ORANGE, "error": AQUA}))
    p.append(series("Trending-query decisions / min", [target(
        "sum by (outcome) (rate(nrt_trend_outcomes_total[5m])) * 60", "{{outcome}}")],
        "none", 16, y, w=8, desc="Only cached_nrt cost an LLM call; cached / confident were free skips"))
    y += 8

    p.append(row("Platform and saturation", y))
    y += 1
    p.append(series("HTTP 5xx ratio by service", [target("http:error_5xx_ratio:5m", "{{job}}")],
                    "percentunit", 0, y, w=8, colors={"search_service": BLUE, "genai_gateway": ORANGE},
                    threshold_lines=((0.05, CRITICAL),)))
    p.append(series("Kafka consumer lag (messages)", [target(
        'kafka:consumer_lag:sum{redpanda_group=~"nrt-trends|catalog-indexer|lake-sink"}',
        "{{redpanda_group}} / {{redpanda_topic}}")], "short", 8, y, w=8,
        threshold_lines=((1000, WARNING),), desc="lake-sink drains once per pipeline run, so it grows by design"))
    p.append(series("Process CPU (cores)", [target("rate(app_process_cpu_seconds_total[5m])", "{{service}}")],
                    "none", 16, y, w=8, threshold_lines=((0.9, WARNING),),
                    colors={"search_service": BLUE, "genai_gateway": ORANGE, "nrt_detector": AQUA}))
    y += 8
    p.append(series("Redis memory", [target("redis_memory_used_bytes", "used")], "bytes", 0, y, w=8,
                    threshold_lines=((1.5e9, WARNING),)))
    p.append(series("Redpanda memory available", [target(
        "redpanda_memory_available_memory / (redpanda_memory_available_memory + redpanda_memory_allocated_memory)",
        "available")], "percentunit", 8, y, w=8, max_=1, threshold_lines=((0.10, WARNING),)))
    p.append({
        "id": _next_id(), "type": "stat", "title": "Scrape targets up", "datasource": DS,
        "gridPos": {"x": 16, "y": y, "w": 8, "h": 8}, "targets": [target("up", "{{job}}")],
        "fieldConfig": {"defaults": {"mappings": [{"type": "value", "options": {
            "0": {"text": "down", "color": CRITICAL}, "1": {"text": "up", "color": GOOD}}}],
            "color": {"mode": "thresholds"}, "thresholds": thresholds((None, CRITICAL), (1, GOOD))},
            "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background",
                    "textMode": "value_and_name", "orientation": "horizontal"},
    })

    return {
        "uid": "intelligent-search", "title": "Intelligent Search", "tags": ["search", "case-study"],
        "timezone": "browser", "schemaVersion": 39, "version": 1, "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"}, "editable": False, "panels": p,
        "annotations": {"list": [{"name": "Firing alerts", "datasource": DS, "enable": True,
                                  "expr": 'ALERTS{alertstate="firing"}', "iconColor": CRITICAL,
                                  "titleFormat": "{{alertname}}", "step": "60s"}]},
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dashboard = build()
    OUT.write_text(json.dumps(dashboard, indent=1) + "\n")
    print(f"wrote {OUT} ({len(dashboard['panels'])} panels)")
