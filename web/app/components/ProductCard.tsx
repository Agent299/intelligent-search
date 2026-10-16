"use client";

import { useState } from "react";
import { formatPrice, type Product } from "@/lib/search";

type EventType = "click" | "add_to_cart" | "purchase";

// Interaction events feed CTR / conversion metrics and the session re-rank (POST /v1/events, via the rewrite in
// next.config.ts). keepalive lets the request finish if the shopper navigates away.
function track(searchId: string, type: EventType, productId: string) {
  fetch("/api/v1/events", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ searchId, type, productId }),
    keepalive: true,
  }).catch(() => {});
}

export function ProductCard({ product, searchId, compact = false }: {
  product: Product;
  searchId: string;
  compact?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<"" | "In cart" | "Purchased">("");

  const view = () => {
    if (!open) track(searchId, "click", product.productId);
    setOpen((o) => !o);
  };

  return (
    <article className={compact ? "card card-compact" : "card"}>
      <button className="card-main" onClick={view} aria-expanded={open}>
        <div className="card-image">
          {product.imageUrl ? (
            // Product images are remote catalog URLs; next/image would need every host allow-listed.
            // eslint-disable-next-line @next/next/no-img-element
            <img src={product.imageUrl} alt="" loading="lazy" />
          ) : (
            <span className="card-noimage">No image</span>
          )}
        </div>
        <h3 className="card-title">{product.title}</h3>
      </button>
      <div className="card-meta">
        <span className="card-price">{formatPrice(product.price)}</span>
        {product.rating != null ? <span className="card-rating">★ {product.rating.toFixed(1)}</span> : null}
      </div>
      {product.brand ? <p className="card-brand">{product.brand}</p> : null}
      {open && !compact ? <p className="card-desc">{product.description || "No description."}</p> : null}
      {compact ? null : (
        <div className="card-actions">
          <button
            disabled={status !== ""}
            onClick={() => { track(searchId, "add_to_cart", product.productId); setStatus("In cart"); }}
          >
            Add to cart
          </button>
          <button
            className="secondary"
            disabled={status === "Purchased"}
            onClick={() => { track(searchId, "purchase", product.productId); setStatus("Purchased"); }}
          >
            Buy now
          </button>
          {status ? <span className="card-status" role="status">{status}</span> : null}
        </div>
      )}
    </article>
  );
}
