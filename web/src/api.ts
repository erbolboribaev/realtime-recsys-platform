// API client for the serving layer. Falls back to the in-browser mock when the page is not
// served next to the backend (design preview, offline demo) or when the URL has ?mock=1.
import { mockApi } from "./mock.ts";
import { HttpError } from "./types.ts";
import type { Api, Item, Stats } from "./types.ts";

interface RecommendResponse {
  items: string[];
  request_id: string;
  variant: string;
  latency_ms: number;
  explain_url?: string;
}

interface ExplanationEvent {
  text: string;
  source: string;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) throw new HttpError(`${init?.method ?? "GET"} ${path} failed with ${res.status}`, res.status);
  return (await res.json()) as T;
}

const httpApi: Api = {
  mock: false,
  // GET /recommend answers with item ids in ranked order plus request_id, variant, latency_ms.
  async recommend(userId, k = 9) {
    const body = await json<RecommendResponse>(`/recommend?user_id=${encodeURIComponent(userId)}&k=${k}&explain=true`);
    return {
      items: body.items.map((id) => ({ item_id: id })),
      variant: body.variant,
      requestId: body.explain_url ? body.request_id : null,
      attributionId: body.request_id,
      latencyMs: Math.round(body.latency_ms),
    };
  },
  async getItems(ids) {
    if (!ids.length) return [];
    return (await json<{ items: Item[] }>(`/items?ids=${ids.map(encodeURIComponent).join(",")}`)).items;
  },
  // The request id ties the click to the list that produced it.
  async postAction(userId, itemId, type, requestId) {
    const res = await fetch("/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, item_id: itemId, event_type: type, request_id: requestId ?? null }),
    });
    if (!res.ok) throw new HttpError(`POST /events failed with ${res.status}`, res.status);
  },
  getStats(token) {
    return json<Stats>("/admin/stats", { headers: token ? { "X-Admin-Token": token } : {} });
  },
  // SSE: "explanation" carries {text, source}; "timeout" means nothing arrived and no fallback exists.
  streamExplanation(requestId, { onText, onDone, onError }) {
    if (!requestId) {
      onError?.(new Error("explanations are off"));
      return () => {};
    }
    const es = new EventSource(`/explain/${encodeURIComponent(requestId)}`);
    es.addEventListener("explanation", (e) => {
      es.close();
      const data = JSON.parse((e as MessageEvent<string>).data) as ExplanationEvent;
      onText(data.text);
      onDone({ source: data.source });
    });
    es.addEventListener("timeout", () => {
      es.close();
      onError?.(new Error("timeout"));
    });
    es.onerror = () => {
      es.close();
      onError?.(new Error("stream error"));
    };
    return () => es.close();
  },
};

const local = /^(localhost|127\.|\[::1\])/.test(location.hostname);
const forceMock = new URLSearchParams(location.search).has("mock");

export const api: Api = local && !forceMock ? httpApi : mockApi;
