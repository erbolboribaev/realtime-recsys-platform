import "./styles.css";
import { api } from "./api.ts";
import { initChrome } from "./chrome.ts";
import { $, h } from "./dom.ts";
import { clock } from "./format.ts";
import { applyStatic, t } from "./i18n.ts";
import type { TextKey } from "./i18n.ts";
import { MIN_LISTS_FOR_COMPARISON, ago, answerShares, lift, serviceState, sparkPath, spendPercent, variantLabel } from "./stats.ts";
import { HttpError } from "./types.ts";
import type { ServiceName, Stats } from "./types.ts";

const POLL_MS = 3000;
const HISTORY = 40;
function stored(): string | null { try { return sessionStorage.getItem("adminToken"); } catch { return null; } }
function remember(value: string): void { try { sessionStorage.setItem("adminToken", value); } catch { /* private mode: the token lives in memory only */ } }
let token: string | null = new URLSearchParams(location.search).get("token") || stored();
if (token) {
  remember(token);
  if (location.search.includes("token=")) history.replaceState(null, "", location.pathname); // keep it out of the address bar
}
let submitted = false;
const series = { rpm: [] as number[], p95: [] as (number | null)[] };
let latest: Stats | null = null;
let paused = false;
let timer: ReturnType<typeof setInterval> | undefined;

const SERVICES: [ServiceName, TextKey][] = [
  ["api", "svc.api"],
  ["redis", "svc.redis"],
  ["feature_job", "svc.feature_job"],
  ["explain_worker", "svc.explain_worker"],
];

function svg(tag: string, attrs: Record<string, string | number> = {}): SVGElement {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
  return el;
}

function sparkline(values: (number | null)[], label: string): SVGElement {
  const { line, area, last } = sparkPath(values, 240, 44);
  const el = svg("svg", { class: "spark", viewBox: "0 0 240 44", preserveAspectRatio: "xMinYMid meet", role: "img", "aria-label": label });
  if (line && last) {
    el.append(svg("path", { class: "area", d: area }), svg("path", { class: "line", d: line }), svg("circle", { class: "end", cx: last[0], cy: last[1], r: 4 }));
  }
  return el;
}

function renderServices(stats: Stats): void {
  $("svc").replaceChildren(...SERVICES.map(([key, label]) => {
    const service = stats.services[key];
    const state = serviceState(service);
    const since = ago(service?.seconds_since_beat);
    const detail = state === "up" ? (since ? `${t("adm.lastSignal")} ${since === "now" ? t("adm.now") : since}` : t("svc.up")) : state === "down" ? t("svc.down") : t("svc.unknown");
    return h("li", { class: `svc ${state}` },
      h("div", { class: "svc-top" }, h("span", { class: "dot", "aria-hidden": "true" }), h("span", { text: t(label) })),
      h("span", { class: "svc-state", text: detail }));
  }));
}

function kpi(label: string, value: string, unit: string | null, sub: string | null, extra?: Node | null): HTMLElement {
  return h("div", { class: "kpi" },
    h("span", { class: "kpi-label", text: label }),
    h("span", { class: "kpi-value" }, value, unit ? h("small", { text: unit }) : null),
    sub ? h("span", { class: "kpi-sub", text: sub }) : null,
    extra ?? null);
}

function renderKpis(stats: Stats): void {
  const tr = stats.traffic;
  const rpm = tr.requests_last_minute;
  const { p95, p50 } = tr.latency_ms;
  const llm = stats.llm;
  const pct = llm ? spendPercent(llm.spent_usd, llm.budget_usd) : 0;
  const meter = h("div", { class: `meter${pct >= 90 ? " warn" : ""}`, role: "img", "aria-label": `${pct}%` }, h("b", { style: `width:${pct}%` }));
  $("kpi").replaceChildren(
    kpi(t("kpi.rpm"), String(rpm), null, `${t("kpi.window")}`, sparkline(series.rpm, t("kpi.rpm"))),
    kpi(t("kpi.p95"), p95 == null ? "–" : String(p95), p95 == null ? null : "ms", p50 == null ? null : `${t("kpi.typical")} ${p50} ms`, sparkline(series.p95, t("kpi.p95"))),
    kpi(t("kpi.events"), String(stats.events?.applied ?? 0), null, t("kpi.eventsSub")),
    kpi(t("kpi.spend"), llm ? `$${llm.spent_usd.toFixed(2)}` : "–", null, llm ? `${t("kpi.of")} $${llm.budget_usd.toFixed(2)}` : t("svc.unknown"), llm ? meter : null));
}

function renderAb(stats: Stats): void {
  const variants = stats.traffic.variants;
  const order = ["control", "treatment", ...Object.keys(variants).filter((k) => k !== "control" && k !== "treatment")].filter((k) => variants[k]);
  const max = Math.max(...order.map((k) => variants[k]?.rate ?? 0), 0.0001);
  if (!order.length) {
    $("ab").replaceChildren(h("p", { class: "muted", text: t("adm.noLists") }));
    $("ab-verdict").textContent = "";
    return;
  }
  $("ab").replaceChildren(...order.map((name) => {
    const v = variants[name] ?? { lists: 0, actions: 0, rate: 0 };
    return h("div", { class: "ab-row" },
      h("div", { class: "ab-name" }, h("span", { text: variantLabel(name) }), h("small", { text: `${v.lists} ${t("adm.lists")} · ${v.actions} ${t("adm.actions")}` })),
      h("div", { class: "bar", role: "img", "aria-label": `${(v.rate * 100).toFixed(1)}%` }, h("b", { style: `width:${Math.round((v.rate / max) * 100)}%` })),
      h("span", { class: "ab-rate", text: `${(v.rate * 100).toFixed(1)}%` }));
  }));
  const diff = lift(variants);
  $("ab-verdict").textContent = diff == null
    ? `${t("adm.wait")} ${MIN_LISTS_FOR_COMPARISON} ${t("adm.waitTail")}`
    : `${t("adm.lift")}: ${diff > 0 ? "+" : ""}${diff}%`;
}

function renderAi(stats: Stats): void {
  const llm = stats.llm;
  const box = $("ai");
  if (!llm) { box.replaceChildren(h("p", { class: "muted", text: t("svc.unknown") })); return; }
  const shares = answerShares(llm.answers);
  if (!shares.total) { box.replaceChildren(h("p", { class: "muted", text: t("adm.noAnswers") })); return; }
  const rows: { key: "ai" | "template" | "from_cache"; cls: string; color: string; label: TextKey }[] = [
    { key: "ai", cls: "s-ai", color: "accent", label: "chat.llm" },
    { key: "template", cls: "s-template", color: "ink", label: "chat.tpl" },
    { key: "from_cache", cls: "s-cache", color: "line", label: "adm.cache" },
  ];
  box.replaceChildren(
    h("div", { class: "stack", role: "img", "aria-label": rows.map((r) => `${t(r.label)} ${shares[r.key]}%`).join(", ") },
      ...rows.map((r) => h("b", { class: r.cls, style: `width:${shares[r.key]}%` }))),
    h("ul", { class: "legend" }, ...rows.map((r) =>
      h("li", {}, h("i", { class: r.cls, style: `background:var(--${r.color})` }), h("span", { text: t(r.label) }), h("span", { text: String(llm.answers[r.key] ?? 0) })))));
}

function render(stats: Stats): void {
  renderServices(stats);
  renderKpis(stats);
  renderAb(stats);
  renderAi(stats);
  const mode = api.mock ? ` · ${t("mock.banner")}` : "";
  $("adm-updated").textContent = `${t("adm.updated")} ${clock(new Date())}${mode} · ${t("adm.every")}`;
}

function notice(key: TextKey | null): void {
  const el = $("adm-notice");
  el.hidden = !key;
  if (key) el.textContent = t(key);
}

async function poll(): Promise<void> {
  try {
    const stats = await api.getStats(token);
    latest = stats;
    series.rpm.push(stats.traffic.requests_last_minute);
    series.p95.push(stats.traffic.latency_ms.p95);
    for (const s of Object.values(series)) while (s.length > HISTORY) s.shift();
    notice(null);
    $("login").hidden = true;
    $("data").hidden = false;
    render(stats);
  } catch (err) {
    const denied = err instanceof HttpError && err.status === 401;
    if (!denied) console.error(err);
    if (denied) {
      clearInterval(timer);
      $("data").hidden = true;
      $("login").hidden = false;
      notice(submitted ? "adm.badToken" : "adm.denied");
      $("token").focus();
    } else {
      notice("adm.error");
    }
  }
}

function schedule(): void {
  clearInterval(timer);
  if (!paused) timer = setInterval(() => void poll(), POLL_MS);
}

$("login").addEventListener("submit", (e) => {
  e.preventDefault();
  token = $<HTMLInputElement>("token").value.trim();
  submitted = true;
  remember(token);
  void poll().then(schedule);
});

$("adm-pause").addEventListener("click", (e) => {
  const button = e.currentTarget as HTMLButtonElement;
  paused = !paused;
  button.setAttribute("aria-pressed", String(paused));
  button.textContent = t(paused ? "adm.resume" : "adm.pause");
  schedule();
  if (!paused) void poll();
});

initChrome({
  onLang: () => {
    $("adm-pause").textContent = t(paused ? "adm.resume" : "adm.pause");
    if (latest) render(latest);
  }
});
applyStatic();
void poll();
schedule();
