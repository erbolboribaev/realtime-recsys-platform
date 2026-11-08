import { api } from "./api.js";
import { categoryHue, clock, formatPrice, pickRandom, shopperIds, shopperName } from "./format.js";
import { applyStatic, setLang, t } from "./i18n.js";
import { iconSvg } from "./icons.js";

const $ = (id) => document.getElementById(id);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const SHOPPERS = shopperIds();
const ACTIONS = [["view", "card.view", "act.view"], ["add_to_cart", "card.cart", "act.cart"], ["purchase", "card.buy", "act.buy"]];

const state = { userId: SHOPPERS[0], items: [], info: new Map(), shown: new Set(), activity: [], events: 0, stopStream: null, busy: false, attributionId: null, sourceKey: "src.none" };

function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style") el.style.cssText = v;
    else el.setAttribute(k, v);
  }
  kids.flat().forEach((c) => c != null && el.append(c));
  return el;
}

/* ---------- static sections ---------- */
function renderStatic() {
  $("steps").replaceChildren(...[1, 2, 3].map((n) => h("li", { class: "step" }, h("h3", { "data-i18n": `how.${n}t` }), h("p", { "data-i18n": `how.${n}d` }))));
  $("shopper").replaceChildren(...SHOPPERS.map((id) => h("option", { value: id, text: shopperName(id) })));
}

/* ---------- cards ---------- */
function renderCards(newIds = new Set()) {
  const cards = state.items.map((rec, i) => {
    const info = state.info.get(rec.item_id) ?? { name: rec.item_id, category: "", price: 0 };
    return h("article", { class: "card" },
      h("div", { class: "cover", style: `--h:${categoryHue(info.category)}` },
        Object.assign(h("span"), { innerHTML: iconSvg(info.category) }).firstChild,
        newIds.has(rec.item_id) ? h("span", { class: "new", text: t("card.new") }) : null,
        h("span", { class: "rank", text: String(i + 1), "aria-label": `#${i + 1}` })),
      h("div", { class: "card-body" },
        h("p", { class: "cat", text: info.category }),
        h("h3", { text: info.name }),
        h("p", { class: "price", text: formatPrice(info.price) }),
        h("div", { class: "acts-row" }, ACTIONS.map(([type, label]) =>
          h("button", { type: "button", class: `act${type === "purchase" ? " buy" : ""}`, text: t(label),
            "aria-label": `${t(label)}: ${info.name}`, onclick: () => act(rec.item_id, type) })))));
  });
  $("cards").replaceChildren(...cards);
  $("cards").setAttribute("aria-busy", "false");
}

function renderSkeleton() {
  $("cards").setAttribute("aria-busy", "true");
  $("cards").replaceChildren(...Array.from({ length: 9 }, () => h("div", { class: "skel" })));
}

/* ---------- pulse, status, activity ---------- */
function setPulse(id, text) {
  const el = $(id); if (!el) return; el.textContent = text;
  if (reduceMotion) return;
  el.classList.add("tick"); setTimeout(() => el.classList.remove("tick"), 500);
}
function setStatus(text, error = false) { const s = $("status"); s.textContent = text; s.classList.toggle("err", error); }

function renderActivity() {
  const list = $("acts");
  if (!state.activity.length) { list.replaceChildren(h("li", { class: "empty", text: t("act.empty") })); return; }
  list.replaceChildren(...state.activity.slice(0, 6).map((a) =>
    h("li", {}, h("span", { text: `${t(a.label)}: ${a.name}` }), h("time", { text: a.time }))));
}

/* ---------- chat (explanation) ---------- */
function bubble(cls, ...kids) { return h("div", { class: `bubble ${cls}` }, ...kids); }
function startExplanation(requestId) {
  state.stopStream?.();
  const chat = $("chat");
  const typing = h("span", { class: "typing", "aria-label": t("chat.wait") }, h("i"), h("i"), h("i"));
  const text = h("span");
  const ai = bubble("ai", text, typing);
  const meBubble = bubble("me", t("chat.ask"), h("time", { text: clock() }));
  chat.replaceChildren(meBubble, ai);
  $("chat-sub").textContent = `${shopperName(state.userId)}`;
  setPulse("p-source", t("src.none"));
  state.stopStream = api.streamExplanation(requestId, {
    onText: (full) => { text.textContent = full; typing.remove(); },
    onDone: ({ source }) => {
      typing.remove();
      const isLlm = source === "llm";
      ai.append(h("span", { class: `src${isLlm ? "" : " tpl"}`, text: t(isLlm ? "chat.llm" : "chat.tpl") }), h("time", { text: clock() }));
      state.sourceKey = isLlm ? "src.llm" : "src.template";
      setPulse("p-source", t(state.sourceKey));
    },
    onError: () => {
      typing.remove();
      text.textContent = t("chat.wait");
    }
  });
}

/* ---------- data flow ---------- */
async function load({ quiet = false } = {}) {
  if (state.busy) return;
  state.busy = true;
  if (!quiet) { renderSkeleton(); setStatus(t("shop.loading")); }
  try {
    const rec = await api.recommend(state.userId, 9);
    const missing = rec.items.map((x) => x.item_id).filter((id) => !state.info.has(id));
    (await api.getItems(missing)).forEach((it) => state.info.set(it.item_id, it));
    const fresh = new Set(rec.items.map((x) => x.item_id).filter((id) => state.shown.size && !state.shown.has(id)));
    state.items = rec.items;
    state.attributionId = rec.attributionId ?? rec.requestId;
    state.shown = new Set(rec.items.map((x) => x.item_id));
    renderCards(fresh);
    setPulse("p-latency", `${rec.latencyMs} ms`);
    setPulse("p-variant", rec.variant);
    setStatus(`${t("shop.updated")} ${clock()}${api.mock ? ` · ${t("mock.banner")}` : ""}`);
    startExplanation(rec.requestId);
  } catch (err) {
    console.error(err);
    $("cards").setAttribute("aria-busy", "false");
    $("cards").replaceChildren();
    setStatus(t("shop.error"), true);
  } finally {
    state.busy = false;
  }
}

async function act(itemId, type) {
  const info = state.info.get(itemId);
  const label = ACTIONS.find((a) => a[0] === type)[2];
  state.activity.unshift({ label, name: info?.name ?? itemId, time: clock() });
  state.events += 1;
  setPulse("p-events", String(state.events));
  renderActivity();
  try { await api.postAction(state.userId, itemId, type, state.attributionId); } catch (err) { console.error(err); setStatus(t("shop.error"), true); return; }
  setTimeout(() => load({ quiet: true }), reduceMotion ? 300 : 2200);
}

function switchShopper(id) {
  state.userId = id; state.shown = new Set(); state.activity = [];
  $("shopper").value = id;
  renderActivity();
  load();
}

/* ---------- chrome: language, theme ---------- */
function safe(fn) { try { return fn(); } catch { return null; } }
function initChrome() {
  document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => {
    setLang(b.dataset.lang); safe(() => localStorage.setItem("lang", b.dataset.lang));
    document.querySelectorAll("[data-lang]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    applyStatic(); renderActivity(); if (state.items.length) renderCards(); setPulse("p-source", t(state.sourceKey));
  }));
  $("theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next; safe(() => localStorage.setItem("theme", next));
  });
  const savedTheme = safe(() => localStorage.getItem("theme")); if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  const savedLang = safe(() => localStorage.getItem("lang"));
  if (savedLang === "uz") document.querySelector('[data-lang="uz"]').click();
}

renderStatic();
setLang("en");
applyStatic();
initChrome();
renderActivity();
$("shopper").addEventListener("change", (e) => switchShopper(e.target.value));
$("random").addEventListener("click", () => switchShopper(pickRandom(SHOPPERS, state.userId)));
$("refresh").addEventListener("click", () => load());
load();
