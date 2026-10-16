// Search API contract (services/search_service, libs/common/src/common/schemas.py) and URL <-> request mapping.
// All storefront state lives in the URL, so every search is a shareable link.

export const SEARCH_API = process.env.SEARCH_API ?? "http://localhost:8000";
export const SESSION_COOKIE = "sid";

export type Product = {
  productId: string;
  title: string;
  description: string;
  imageUrl?: string;
  price: { amount: number; currency: string };
  brand?: string;
  rating?: number;
};

export type FacetValue = { value: string; label: string; count: number };

export type SearchDebug = {
  servedBy: "results_cache" | "l1" | "l2" | "warm" | "fast_path" | "fallback";
  confidence?: number;
  latencyMs: number;
  fastPathConfidence?: number;
  rewrittenQuery?: string;
  cachedQuery?: string;
  cacheSimilarity?: number;
  ranker?: string;
};

export type SearchResponse = {
  searchId: string;
  correctedQuery?: string;
  results: Product[];
  relatedSearches: string[];
  pairsWellWith: Product[];
  facets?: { brands: FacetValue[]; categories: FacetValue[]; priceMin?: number; priceMax?: number };
  pagination: { currentPage: number; totalPages: number; totalResults: number };
  debug?: SearchDebug;
};

export type Sort = "relevance" | "price_asc" | "price_desc";

export type SearchState = {
  q: string;
  brands: string[];
  category?: string;
  priceMin?: string;
  priceMax?: string;
  sort: Sort;
  page: number;
};

type Params = Record<string, string | string[] | undefined>;

const first = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v);
const all = (v: string | string[] | undefined) => (v === undefined ? [] : Array.isArray(v) ? v : [v]);
const SORTS: Sort[] = ["relevance", "price_asc", "price_desc"];

export function readState(params: Params): SearchState {
  const sort = first(params.sort) as Sort;
  const page = Number(first(params.page));
  return {
    q: (first(params.q) ?? "").trim(),
    brands: all(params.brand),
    category: first(params.category) || undefined,
    priceMin: first(params.pmin) || undefined,
    priceMax: first(params.pmax) || undefined,
    sort: SORTS.includes(sort) ? sort : "relevance",
    page: Number.isInteger(page) && page >= 1 && page <= 50 ? page : 1,
  };
}

/** Link to the same search with some state changed; any filter or sort change goes back to page 1. */
export function href(state: SearchState, change: Partial<SearchState> = {}): string {
  const s = { ...state, page: 1, ...change };
  const p = new URLSearchParams({ q: s.q });
  s.brands.forEach((b) => p.append("brand", b));
  if (s.category) p.set("category", s.category);
  if (s.priceMin) p.set("pmin", s.priceMin);
  if (s.priceMax) p.set("pmax", s.priceMax);
  if (s.sort !== "relevance") p.set("sort", s.sort);
  if (s.page > 1) p.set("page", String(s.page));
  return `/?${p}`;
}

export function hasFilters(s: SearchState): boolean {
  return s.brands.length > 0 || !!s.category || !!s.priceMin || !!s.priceMax;
}

export class SearchError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function search(state: SearchState, sessionId?: string): Promise<SearchResponse> {
  const filters: Record<string, unknown> = {};
  if (state.brands.length) filters.brand = state.brands;
  if (state.category) filters.category = state.category;
  if (state.priceMin) filters.price_min = state.priceMin;
  if (state.priceMax) filters.price_max = state.priceMax;
  const p = new URLSearchParams({
    q: state.q, page: String(state.page), pageSize: "24", sortBy: state.sort, device: "web", debug: "true",
  });
  if (Object.keys(filters).length) p.set("filters", JSON.stringify(filters));
  if (sessionId) p.set("sessionId", sessionId);
  const res = await fetch(`${SEARCH_API}/v1/search?${p}`, { signal: AbortSignal.timeout(10_000) });
  if (!res.ok) {
    const detail = await res.json().then((b) => b.detail, () => undefined);
    throw new SearchError(res.status, typeof detail === "string" ? detail : `search failed (${res.status})`);
  }
  return res.json();
}

export function formatPrice({ amount, currency }: Product["price"]): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency }).format(amount);
}
