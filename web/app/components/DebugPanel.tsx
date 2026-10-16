import type { SearchDebug } from "@/lib/search";

// Which serving path answered, and why: the case study's architecture made visible (plan Phase 12).
const PATHS: Record<SearchDebug["servedBy"], { label: string; explain: string }> = {
  results_cache: { label: "Results cache", explain: "Hot query: the final ranked ids came straight from Redis." },
  l1: { label: "L1 cache", explain: "In-process cache of an LLM-enriched query compiled offline (Phase 5–6)." },
  l2: { label: "L2 cache", explain: "Redis cache of an LLM-enriched query compiled offline (Phase 5–6)." },
  warm: { label: "Warm tier", explain: "A semantically similar cached query answered this one, with no LLM call." },
  fast_path: { label: "Fast path", explain: "Hybrid BM25 + k-NN retrieval, fused with RRF; confident enough on its own." },
  fallback: { label: "LLM fallback", explain: "Low confidence: Haiku rewrote the query, then retrieval ran again." },
};

const fmt = (n?: number, digits = 2) => (n == null ? "—" : n.toFixed(digits));

export function DebugPanel({ debug, searchId }: { debug: SearchDebug; searchId: string }) {
  const path = PATHS[debug.servedBy];
  const rows: [string, string][] = [
    ["Latency", `${debug.latencyMs.toFixed(1)} ms (server)`],
    ["Confidence", fmt(debug.confidence)],
  ];
  if (debug.fastPathConfidence != null) rows.push(["Fast-path confidence", fmt(debug.fastPathConfidence)]);
  if (debug.rewrittenQuery) rows.push(["LLM rewrite", debug.rewrittenQuery]);
  if (debug.cachedQuery) rows.push(["Cached query used", debug.cachedQuery]);
  if (debug.cacheSimilarity != null) rows.push(["Similarity", fmt(debug.cacheSimilarity, 3)]);
  rows.push(["Re-rank", debug.ranker ?? "none"], ["Search id", searchId]);

  return (
    <details className="debug">
      <summary>
        <span className={`path path-${debug.servedBy}`}>{path.label}</span>
        <span className="debug-latency">{debug.latencyMs.toFixed(0)} ms</span>
        <span className="debug-hint">How was this served?</span>
      </summary>
      <p>{path.explain}</p>
      <dl>
        {rows.map(([k, v]) => (
          <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
        ))}
      </dl>
    </details>
  );
}
