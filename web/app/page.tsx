import Link from "next/link";
import { cookies } from "next/headers";
import { Suspense } from "react";
import { DebugPanel } from "@/app/components/DebugPanel";
import { Facets } from "@/app/components/Facets";
import { ProductCard } from "@/app/components/ProductCard";
import { href, readState, search, SearchError, SESSION_COOKIE, type SearchState, type Sort } from "@/lib/search";

// The case study's own example queries (doc p.181), as one-click starting points.
const EXAMPLES = [
  "healthy snacks without nuts for kids", "post-workout food", "stuff for my dorm party",
  "a summer dress that's not too formal", "organic aged basmati rice", "old people home footwear", "brownie",
];

const SORT_LABELS: Record<Sort, string> = {
  relevance: "Relevance", price_asc: "Price: low to high", price_desc: "Price: high to low",
};

export default function Page({ searchParams }: PageProps<"/">) {
  return (
    <main>
      <Suspense fallback={<SearchBar />}>
        <SearchBarFromUrl searchParams={searchParams} />
      </Suspense>
      <Suspense fallback={<p className="loading" role="status">Searching…</p>}>
        <Results searchParams={searchParams} />
      </Suspense>
    </main>
  );
}

function SearchBar({ q = "" }: { q?: string }) {
  return (
    <form action="/" className="searchbar" role="search">
      <input name="q" type="search" defaultValue={q} placeholder="Search groceries, clothing, home…"
        aria-label="Search" autoComplete="off" maxLength={200} />
      <button type="submit">Search</button>
    </form>
  );
}

async function SearchBarFromUrl({ searchParams }: Pick<PageProps<"/">, "searchParams">) {
  const { q } = readState(await searchParams);
  return <SearchBar key={q} q={q} />;
}

async function Results({ searchParams }: Pick<PageProps<"/">, "searchParams">) {
  const state = readState(await searchParams);
  if (!state.q) return <Welcome />;

  const sessionId = (await cookies()).get(SESSION_COOKIE)?.value;
  let res;
  try {
    res = await search(state, sessionId);
  } catch (e) {
    const message = e instanceof SearchError && e.status === 422 ? e.message
      : "The search service is unavailable. Is it running (make search)?";
    return <p className="notice error" role="alert">{message}</p>;
  }

  const { pagination: pg } = res;
  return (
    <>
      {res.correctedQuery ? (
        <p className="notice">
          Showing results for <strong>{res.correctedQuery}</strong>. You searched for “{state.q}”.
        </p>
      ) : null}

      {res.relatedSearches.length ? (
        <nav className="related" aria-label="Related searches">
          <span>Related:</span>
          {res.relatedSearches.map((r) => (
            <Link key={r} prefetch={false} href={href({ ...state, brands: [], category: undefined }, { q: r })}>{r}</Link>
          ))}
        </nav>
      ) : null}

      <div className="layout">
        <Facets state={state} facets={res.facets} />
        <section className="results" aria-label="Results">
          <div className="toolbar">
            <p>{pg.totalResults ? `${pg.totalResults} results` : "No results"}</p>
            <nav className="sort" aria-label="Sort">
              {(Object.keys(SORT_LABELS) as Sort[]).map((s) => (
                <Link key={s} prefetch={false} href={href(state, { sort: s })}
                  aria-current={state.sort === s ? "true" : undefined}>
                  {SORT_LABELS[s]}
                </Link>
              ))}
            </nav>
          </div>
          {res.debug ? <DebugPanel debug={res.debug} searchId={res.searchId} /> : null}

          {res.results.length ? (
            <div className="grid">
              {res.results.map((p) => <ProductCard key={p.productId} product={p} searchId={res.searchId} />)}
            </div>
          ) : (
            <NoResults state={state} />
          )}

          {pg.totalPages > 1 ? <Pager state={state} totalPages={pg.totalPages} /> : null}

          {res.pairsWellWith.length ? (
            <section className="pairs" aria-label="Pairs well with">
              <h2>Pairs well with</h2>
              <div className="rail">
                {res.pairsWellWith.map((p) => (
                  <ProductCard key={p.productId} product={p} searchId={res.searchId} compact />
                ))}
              </div>
            </section>
          ) : null}
        </section>
      </div>
    </>
  );
}

function Pager({ state, totalPages }: { state: SearchState; totalPages: number }) {
  const last = Math.min(totalPages, 50);
  return (
    <nav className="pager" aria-label="Pages">
      {state.page > 1 ? <Link prefetch={false} href={href(state, { page: state.page - 1 })}>← Previous</Link> : <span />}
      <span>Page {state.page} of {last}</span>
      {state.page < last ? <Link prefetch={false} href={href(state, { page: state.page + 1 })}>Next →</Link> : <span />}
    </nav>
  );
}

function NoResults({ state }: { state: SearchState }) {
  return (
    <div className="empty">
      <p>Nothing matched “{state.q}”.</p>
      <p>Try fewer words, check the spelling, or start from one of the examples below.</p>
      <Examples />
    </div>
  );
}

function Welcome() {
  return (
    <div className="welcome">
      <h1>Search like you would ask a shop assistant</h1>
      <p>
        Plain-language needs, negations and typos all work. Open “How was this served?” on any results page to see
        which path answered: a cache tier, the hybrid fast path, or the LLM fallback.
      </p>
      <Examples />
    </div>
  );
}

function Examples() {
  return (
    <ul className="examples">
      {EXAMPLES.map((q) => (
        <li key={q}><Link prefetch={false} href={href({ q, brands: [], sort: "relevance", page: 1 })}>{q}</Link></li>
      ))}
    </ul>
  );
}
