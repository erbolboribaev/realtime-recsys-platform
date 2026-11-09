import "./styles.css";
import { api } from "./api.ts";
import { initChrome } from "./chrome.ts";
import { $, h } from "./dom.ts";
import { categoryHue, clock, formatPrice, pickRandom, shopperIds, shopperName } from "./format.ts";
import { applyStatic, setLang, t } from "./i18n.ts";
import type { TextKey } from "./i18n.ts";
import { iconSvg } from "./icons.ts";
import type { ActionType, Item } from "./types.ts";

const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const BASE_SHOPPERS = shopperIds();
const requested = new URLSearchParams(location.search).get("shopper"); // lets a link or a test open as a given shopper
const SHOPPERS = requested && !BASE_SHOPPERS.includes(requested) ? [requested, ...BASE_SHOPPERS] : BASE_SHOPPERS;

const ACTIONS: { type: ActionType; button: TextKey; done: TextKey }[] = [
  { type: "view", button: "card.view", done: "act.view" },
  { type: "add_to_cart", button: "card.cart", done: "act.cart" },
  { type: "purchase", button: "card.buy", done: "act.buy" },
];

interface Activity {
  label: TextKey;
  name: string;
  time: string;
}

const state = {
  userId: requested || (SHOPPERS[0] as string),
  items: [] as { item_id: string }[],
  info: new Map<string, Item>(),
  shown: new Set<string>(),
  activity: [] as Activity[],
  stopStream: null as (() => void) | null,
  busy: false,
  attributionId: null as string | null,
};

/* ---------- static sections ---------- */
function renderStatic(): void {
  $("steps").replaceChildren(
    ...[1, 2, 3].map((n) => h("li", { class: "step" }, h("h3", { "data-i18n": `how.${n}t` }), h("p", { "data-i18n": `how.${n}d` }))),
  );
  $("shopper").replaceChildren(...SHOPPERS.map((id) => h("option", { value: id, text: shopperName(id) })));
}

/* ---------- cards ---------- */
function coverIcon(category: string): Node {
  const holder = document.createElement("span");
  holder.innerHTML = iconSvg(category); // our own static markup, never API text
  return holder.firstChild as Node;
}

function renderCards(newIds: Set<string> = new Set()): void {
  const cards = state.items.map((rec, i) => {
    const info = state.info.get(rec.item_id) ?? { item_id: rec.item_id, name: rec.item_id, category: "", price: 0 };
    return h(
      "article",
      { class: "card", "data-item": rec.item_id },
      h(
        "div",
        { class: "cover", style: `--h:${categoryHue(info.category)}` },
        coverIcon(info.category),
        newIds.has(rec.item_id) ? h("span", { class: "new", text: t("card.new") }) : null,
        h("span", { class: "rank", text: String(i + 1), "aria-label": `#${i + 1}` }),
      ),
      h(
        "div",
        { class: "card-body" },
        h("p", { class: "cat", text: info.category }),
        h("h3", { text: info.name }),
        h("p", { class: "price", text: formatPrice(info.price) }),
        h(
          "div",
          { class: "acts-row" },
          ACTIONS.map(({ type, button }) =>
            h("button", {
              type: "button",
              class: `act${type === "purchase" ? " buy" : ""}`,
              text: t(button),
              "aria-label": `${t(button)}: ${info.name}`,
              onclick: () => void act(rec.item_id, type),
            }),
          ),
        ),
      ),
    );
  });
  $("cards").replaceChildren(...cards);
  $("cards").setAttribute("aria-busy", "false");
}

function renderSkeleton(): void {
  $("cards").setAttribute("aria-busy", "true");
  $("cards").replaceChildren(...Array.from({ length: 9 }, () => h("div", { class: "skel" })));
}

/* ---------- status and activity ---------- */
function setStatus(text: string, error = false): void {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("err", error);
}

function renderActivity(): void {
  const list = $("acts");
  if (!state.activity.length) {
    list.replaceChildren(h("li", { class: "empty", text: t("act.empty") }));
    return;
  }
  list.replaceChildren(
    ...state.activity.slice(0, 6).map((a) => h("li", {}, h("span", { text: `${t(a.label)}: ${a.name}` }), h("time", { text: a.time }))),
  );
}

/* ---------- chat (explanation) ---------- */
function bubble(cls: string, ...kids: (Node | string)[]): HTMLElement {
  return h("div", { class: `bubble ${cls}` }, ...kids);
}

function startExplanation(requestId: string | null): void {
  state.stopStream?.();
  const typing = h("span", { class: "typing", "aria-label": t("chat.wait") }, h("i"), h("i"), h("i"));
  const text = h("span");
  const ai = bubble("ai", text, typing);
  $("chat").replaceChildren(bubble("me", t("chat.ask"), h("time", { text: clock() })), ai);
  $("chat-sub").textContent = shopperName(state.userId);
  state.stopStream = api.streamExplanation(requestId, {
    onText: (full) => {
      text.textContent = full;
      typing.remove();
    },
    onDone: ({ source }) => {
      typing.remove();
      const isLlm = source === "llm";
      ai.append(h("span", { class: `src${isLlm ? "" : " tpl"}`, text: t(isLlm ? "chat.llm" : "chat.tpl") }), h("time", { text: clock() }));
    },
    onError: () => {
      typing.remove();
      text.textContent = t("chat.wait");
    },
  });
}

/* ---------- data flow ---------- */
async function load({ quiet = false } = {}): Promise<void> {
  if (state.busy) return;
  state.busy = true;
  if (!quiet) {
    renderSkeleton();
    setStatus(t("shop.loading"));
  }
  try {
    const rec = await api.recommend(state.userId, 9);
    const missing = rec.items.map((x) => x.item_id).filter((id) => !state.info.has(id));
    (await api.getItems(missing)).forEach((it) => state.info.set(it.item_id, it));
    const fresh = new Set(rec.items.map((x) => x.item_id).filter((id) => state.shown.size && !state.shown.has(id)));
    state.items = rec.items;
    state.attributionId = rec.attributionId;
    state.shown = new Set(rec.items.map((x) => x.item_id));
    renderCards(fresh);
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

async function act(itemId: string, type: ActionType): Promise<void> {
  const done = ACTIONS.find((a) => a.type === type)?.done ?? "act.view";
  state.activity.unshift({ label: done, name: state.info.get(itemId)?.name ?? itemId, time: clock() });
  renderActivity();
  try {
    await api.postAction(state.userId, itemId, type, state.attributionId);
  } catch (err) {
    console.error(err);
    setStatus(t("shop.error"), true);
    return;
  }
  setTimeout(() => void load({ quiet: true }), reduceMotion ? 300 : 2200);
}

function switchShopper(id: string): void {
  state.userId = id;
  state.shown = new Set();
  state.activity = [];
  $<HTMLSelectElement>("shopper").value = id;
  renderActivity();
  void load();
}

renderStatic();
setLang("en");
applyStatic();
initChrome({
  onLang: () => {
    renderActivity();
    if (state.items.length) renderCards();
  },
});
renderActivity();
$<HTMLSelectElement>("shopper").value = state.userId;
$<HTMLSelectElement>("shopper").addEventListener("change", (e) => switchShopper((e.target as HTMLSelectElement).value));
$("random").addEventListener("click", () => switchShopper(pickRandom(SHOPPERS, state.userId)));
$("refresh").addEventListener("click", () => void load());
void load();
