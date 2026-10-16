import Link from "next/link";
import { hasFilters, href, type SearchResponse, type SearchState } from "@/lib/search";

// Filter facets as plain links and a GET form: no client JavaScript, and every filtered view is a URL.
// prefetch={false}: each link is a different search, and prefetching would run (and log) all of them.
export function Facets({ state, facets }: { state: SearchState; facets: SearchResponse["facets"] }) {
  const toggleBrand = (b: string) =>
    href(state, { brands: state.brands.includes(b) ? state.brands.filter((x) => x !== b) : [...state.brands, b] });

  return (
    <aside className="facets" aria-label="Filters">
      {hasFilters(state) ? (
        <Link className="clear" prefetch={false}
          href={href(state, { brands: [], category: undefined, priceMin: undefined, priceMax: undefined })}>
          Clear all filters
        </Link>
      ) : null}

      {state.category || facets?.categories.length ? (
        <section>
          <h2>Category</h2>
          {state.category ? (
            <Link className="facet active" prefetch={false} href={href(state, { category: undefined })}>
              ✕ {state.category.split(" > ").at(-1)}
            </Link>
          ) : null}
          {facets?.categories.map((c) => (
            <Link key={c.value} className="facet" prefetch={false} href={href(state, { category: c.value })}
              title={c.value}>
              {c.label} <span className="count">{c.count}</span>
            </Link>
          ))}
        </section>
      ) : null}

      {state.brands.length || facets?.brands.length ? (
        <section>
          <h2>Brand</h2>
          {state.brands.filter((b) => !facets?.brands.some((f) => f.value === b)).map((b) => (
            <Link key={b} className="facet active" prefetch={false} href={toggleBrand(b)}>✕ {b}</Link>
          ))}
          {facets?.brands.map((b) => (
            <Link key={b.value} prefetch={false} href={toggleBrand(b.value)}
              className={state.brands.includes(b.value) ? "facet active" : "facet"}>
              {state.brands.includes(b.value) ? "✕ " : ""}{b.label} <span className="count">{b.count}</span>
            </Link>
          ))}
        </section>
      ) : null}

      <section>
        <h2>Price</h2>
        <form action="/" className="price-form">
          <input type="hidden" name="q" value={state.q} />
          {state.brands.map((b) => <input key={b} type="hidden" name="brand" value={b} />)}
          {state.category ? <input type="hidden" name="category" value={state.category} /> : null}
          {state.sort !== "relevance" ? <input type="hidden" name="sort" value={state.sort} /> : null}
          <input name="pmin" type="number" min="0" step="any" inputMode="decimal" aria-label="Minimum price"
            placeholder={facets?.priceMin != null ? `${Math.floor(facets.priceMin)}` : "Min"}
            defaultValue={state.priceMin} />
          <span>–</span>
          <input name="pmax" type="number" min="0" step="any" inputMode="decimal" aria-label="Maximum price"
            placeholder={facets?.priceMax != null ? `${Math.ceil(facets.priceMax)}` : "Max"}
            defaultValue={state.priceMax} />
          <button type="submit" className="secondary">Go</button>
        </form>
      </section>
    </aside>
  );
}
