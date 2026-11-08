// In-browser stand-in for the API, used when no backend is reachable (design preview, offline demo).
// Same 20 categories as the real catalog, so the demo and the live API look alike.
const CATEGORIES = [
  "Running shoes", "Sports watches", "Backpacks", "Headphones", "Laptops", "Smartphones", "Cameras",
  "Coffee makers", "Cookware", "Books", "Board games", "Yoga mats", "Bicycles", "Sunglasses",
  "Jackets", "Desk lamps", "Keyboards", "Monitors", "Water bottles", "Tents"
];
const SINGULAR = [
  "Running shoe", "Sports watch", "Backpack", "Headphone", "Laptop", "Smartphone", "Camera",
  "Coffee maker", "Cookware set", "Book", "Board game", "Yoga mat", "Bicycle", "Pair of sunglasses",
  "Jacket", "Desk lamp", "Keyboard", "Monitor", "Water bottle", "Tent"
];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = rng(42);
const ITEMS = Array.from({ length: 200 }, (_, i) => {
  const c = i % CATEGORIES.length;
  return {
    item_id: `item_${String(i + 1).padStart(5, "0")}`,
    category: CATEGORIES[c],
    singular: SINGULAR[c],
    name: `${SINGULAR[c]} model ${i}`,
    price: Math.round(40 + rand() * rand() * 1400),
    popularity: rand()
  };
});
const BY_ID = new Map(ITEMS.map((x) => [x.item_id, x]));
const store = new Map(); // userId -> { aff: number[], seen: Set }
const requests = new Map();
const pulse = { tick: 0, control: 120, treatment: 130, cA: 22, tA: 29, events: 840, llm: 60, tpl: 4 };

function userState(userId) {
  if (!store.has(userId)) {
    const n = Number(userId.replace(/\D/g, "")) || 1;
    const aff = new Array(CATEGORIES.length).fill(0);
    aff[n % 20] = 2.5;
    aff[(n * 7 + 3) % 20] += 1.2;
    store.set(userId, { aff, seen: new Set() });
  }
  return store.get(userId);
}

export const mockApi = {
  mock: true,
  async recommend(userId, k = 9) {
    const t0 = performance.now();
    await sleep(60 + Math.random() * 90);
    const u = userState(userId);
    const n = Number(userId.replace(/\D/g, "")) || 1;
    const variant = n % 2 ? "treatment" : "control";
    const noise = rng(n);
    const scored = ITEMS.map((it) => {
      const ci = CATEGORIES.indexOf(it.category);
      const personal = variant === "treatment" ? Math.min(u.aff[ci], 8) * 0.3 : 0;
      return { item_id: it.item_id, score: it.popularity * 2.2 + personal + noise() * 0.4 - (u.seen.has(it.item_id) ? 1.5 : 0) };
    }).sort((a, b) => b.score - a.score);
    const perCat = {};
    const top = scored.filter((x) => { const c = BY_ID.get(x.item_id).category; perCat[c] = (perCat[c] ?? 0) + 1; return perCat[c] <= 3; }).slice(0, k);
    const requestId = `req_${Math.random().toString(36).slice(2, 10)}`;
    requests.set(requestId, { userId, top: top.map((s) => BY_ID.get(s.item_id)), aff: [...u.aff] });
    return { items: top.map((x) => ({ item_id: x.item_id })), variant, requestId, latencyMs: Math.round(performance.now() - t0) };
  },
  async getItems(ids) {
    await sleep(15);
    return ids.map((id) => BY_ID.get(id)).filter(Boolean).map(({ item_id, name, category, price }) => ({ item_id, name, category, price }));
  },
  async postAction(userId, itemId, type) {
    await sleep(40);
    const it = BY_ID.get(itemId);
    const u = userState(userId);
    if (!it) return;
    u.aff[CATEGORIES.indexOf(it.category)] += type === "purchase" ? 3 : type === "add_to_cart" ? 2 : 1;
    u.seen.add(itemId);
  },
  streamExplanation(requestId, { onText, onDone }) {
    let stopped = false;
    const req = requests.get(requestId);
    const run = async () => {
      await sleep(900 + Math.random() * 600);
      if (stopped || !req) return;
      const best = req.aff.indexOf(Math.max(...req.aff));
      const [a, b, c] = req.top;
      const text = `You have been browsing ${CATEGORIES[best].toLowerCase()} lately, so ${a.name}, ${b.name} and ${c.name} sit at the top. They match what you looked at most and are popular with similar shoppers.`;
      const words = text.split(" ");
      let out = "";
      for (const w of words) {
        if (stopped) return;
        out += (out ? " " : "") + w;
        onText(out);
        await sleep(28 + Math.random() * 30);
      }
      onDone({ source: "llm" });
    };
    run();
    return () => { stopped = true; };
  },
  // Slowly moving made-up numbers so the dashboard looks alive without a backend.
  async getStats() {
    await new Promise((r) => setTimeout(r, 40));
    pulse.tick += 1;
    const rpm = Math.max(0, Math.round(26 + Math.sin(pulse.tick / 4) * 9 + Math.random() * 6));
    pulse.control += rpm * 0.45; pulse.treatment += rpm * 0.55;
    pulse.cA += rpm * 0.45 * 0.19; pulse.tA += rpm * 0.55 * 0.23;
    pulse.events += Math.round(rpm * 0.8);
    pulse.llm += 1; pulse.tpl += pulse.tick % 9 === 0 ? 1 : 0;
    const rate = (a, l) => (l ? Math.round((a / l) * 10000) / 10000 : 0);
    return {
      generated_at: new Date().toISOString(),
      services: { api: { alive: true }, redis: { alive: true }, feature_job: { alive: true, seconds_since_beat: 1.2 }, explain_worker: { alive: true, seconds_since_beat: 3.4 } },
      api: { status: "ok" },
      traffic: {
        window_s: 300, requests_last_minute: rpm, requests_in_window: Math.round(pulse.control + pulse.treatment),
        latency_ms: { p50: Math.round(14 + Math.random() * 4), p95: Math.round(38 + Math.sin(pulse.tick / 3) * 8 + Math.random() * 5) },
        variants: {
          control: { lists: Math.round(pulse.control), actions: Math.round(pulse.cA), rate: rate(pulse.cA, pulse.control) },
          treatment: { lists: Math.round(pulse.treatment), actions: Math.round(pulse.tA), rate: rate(pulse.tA, pulse.treatment) }
        }
      },
      llm: { spent_usd: 0.0021 * pulse.llm, budget_usd: 1, answers: { ai: pulse.llm, template: pulse.tpl, from_cache: Math.round(pulse.llm * 0.3) } },
      events: { applied: pulse.events }
    };
  }
};
