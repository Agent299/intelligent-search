from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("gateway_llm_attempts_total", "LLM attempts by outcome", ["task", "model", "outcome"])
TOKENS = Counter("gateway_llm_tokens_total", "Tokens by kind", ["task", "model", "kind"])
COST = Counter("gateway_llm_cost_usd_total", "Estimated spend in USD", ["task", "model"])
LATENCY = Histogram("gateway_llm_latency_seconds", "Per-attempt latency", ["task", "model"],
                    buckets=(0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 30))
ESCALATIONS = Counter("gateway_escalations_total", "Moves to the next model in a chain",
                      ["task", "from_model", "to_model", "reason"])
BREAKER = Gauge("gateway_breaker_state", "0=closed 1=half_open 2=open", ["model"])
