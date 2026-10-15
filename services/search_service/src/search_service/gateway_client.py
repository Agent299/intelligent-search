"""Thin client for the GenAI gateway. Search treats every gateway problem as 'no rewrite available'."""

import logging

import httpx

log = logging.getLogger(__name__)


class GatewayClient:
    def __init__(self, base_url: str, timeout_s: float = 4.5, client: httpx.AsyncClient | None = None):
        # Slightly above the gateway's own 4 s task budget so the gateway, not us, decides failover.
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def rewrite(self, query: str, candidate_categories: list[str]) -> dict | None:
        """Returns the gateway result ({"valid", "output", "model", "cost_usd", ...}) or None on any failure."""
        return await self.run("rewrite", {"query": query, "candidate_categories": candidate_categories})

    async def run(self, task: str, inputs: dict) -> dict | None:
        try:
            resp = await self._client.post(f"/v1/llm/{task}", json={"inputs": inputs})
        except httpx.HTTPError as e:
            log.warning("gateway %s failed: %s", task, type(e).__name__)
            return None
        if resp.status_code != 200:
            log.warning("gateway %s returned HTTP %s", task, resp.status_code)
            return None
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
