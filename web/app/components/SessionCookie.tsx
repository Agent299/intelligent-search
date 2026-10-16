"use client";

import { useEffect } from "react";
import { SESSION_COOKIE } from "@/lib/search";

/** Gives the browser a session id (read on the server for the session-brand re-rank, Phase 8). */
export function SessionCookie() {
  useEffect(() => {
    if (!document.cookie.split("; ").some((c) => c.startsWith(`${SESSION_COOKIE}=`))) {
      document.cookie = `${SESSION_COOKIE}=${crypto.randomUUID()}; path=/; max-age=1800; samesite=lax`;
    }
  }, []);
  return null;
}
