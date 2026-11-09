// The shapes that cross the network. They mirror the serving layer's responses.

export type ActionType = "view" | "add_to_cart" | "purchase";

export interface Item {
  item_id: string;
  name: string;
  category: string;
  price: number;
}

export interface Recommendations {
  /** Item ids in ranked order. */
  items: { item_id: string }[];
  variant: string;
  /** Present only when an explanation will follow. */
  requestId: string | null;
  /** Ties a click to the list that produced it, which A/B analysis relies on. */
  attributionId: string;
  latencyMs: number;
}

export interface ExplanationHandlers {
  onText(fullText: string): void;
  onDone(result: { source: string }): void;
  onError?(error: Error): void;
}

export interface ServiceStatus {
  alive: boolean;
  seconds_since_beat?: number | null;
}

export type ServiceName = "api" | "redis" | "feature_job" | "explain_worker";

export interface VariantStats {
  lists: number;
  actions: number;
  rate: number;
}

export interface Stats {
  generated_at: string;
  services: Partial<Record<ServiceName, ServiceStatus>>;
  api: { status: string };
  traffic: {
    window_s: number;
    requests_last_minute: number;
    requests_in_window: number;
    latency_ms: { p50: number | null; p95: number | null };
    variants: Record<string, VariantStats>;
  };
  llm: {
    spent_usd: number;
    budget_usd: number;
    answers: { ai: number; template: number; from_cache: number };
  } | null;
  events: { applied?: number } | null;
}

export interface Api {
  /** True when the data is made up in the browser. */
  readonly mock: boolean;
  recommend(userId: string, k?: number): Promise<Recommendations>;
  getItems(ids: string[]): Promise<Item[]>;
  postAction(userId: string, itemId: string, type: ActionType, requestId?: string | null): Promise<void>;
  getStats(token?: string | null): Promise<Stats>;
  /** Returns a function that stops the stream. */
  streamExplanation(requestId: string | null, handlers: ExplanationHandlers): () => void;
}

export class HttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "HttpError";
  }
}
