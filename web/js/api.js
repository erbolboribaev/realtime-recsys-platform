// API client for the serving layer. Falls back to the in-browser mock when the page is not
// served next to the backend (design preview, offline demo) or when the URL has ?mock=1.
import { mockApi } from "./mock.js";

const local = /^(localhost|127\.|\[::1\])/.test(location.hostname);
const forceMock = new URLSearchParams(location.search).has("mock");

async function json(path, init) {
  const res = await fetch(path, init);
  if (!res.ok) throw new Error(`${init?.method ?? "GET"} ${path} failed with ${res.status}`);
  return res.json();
}

const httpApi = {
  mock: false,
  // GET /recommend answers with item ids in ranked order plus request_id, variant, latency_ms.
  async recommend(userId, k = 9) {
    const body = await json(`/recommend?user_id=${encodeURIComponent(userId)}&k=${k}&explain=true`);
    return {
      items: body.items.map((id) => ({ item_id: id })),
      variant: body.variant,
      requestId: body.explain_url ? body.request_id : null,
      attributionId: body.request_id,
      latencyMs: Math.round(body.latency_ms)
    };
  },
  async getItems(ids) {
    if (!ids.length) return [];
    return (await json(`/items?ids=${ids.map(encodeURIComponent).join(",")}`)).items;
  },
  // The request id ties the click to the list that produced it, which A/B analysis relies on.
  async postAction(userId, itemId, type, requestId) {
    const res = await fetch("/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, item_id: itemId, event_type: type, request_id: requestId ?? null })
    });
    if (!res.ok) throw new Error(`POST /events failed with ${res.status}`);
  },
  // SSE: "explanation" carries {text, source}; "timeout" means nothing arrived and no fallback exists.
  streamExplanation(requestId, { onText, onDone, onError }) {
    if (!requestId) { onError?.(new Error("explanations are off")); return () => {}; }
    const es = new EventSource(`/explain/${encodeURIComponent(requestId)}`);
    es.addEventListener("explanation", (e) => {
      es.close();
      const data = JSON.parse(e.data);
      onText(data.text);
      onDone({ source: data.source });
    });
    es.addEventListener("timeout", () => { es.close(); onError?.(new Error("timeout")); });
    es.onerror = () => { es.close(); onError?.(new Error("stream error")); };
    return () => es.close();
  }
};

export const api = local && !forceMock ? httpApi : mockApi;
