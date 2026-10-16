import type { NextConfig } from "next";

const SEARCH_API = process.env.SEARCH_API ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  // Browser -> search API for interaction events, same-origin so the API needs no CORS.
  async rewrites() {
    return [{ source: "/api/v1/events", destination: `${SEARCH_API}/v1/events` }];
  },
};

export default nextConfig;
