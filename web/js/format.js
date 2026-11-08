// Pure helpers, covered by node tests.
export function formatPrice(value) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);
}

export function categoryHue(category) {
  let hash = 0;
  for (const ch of String(category)) hash = (hash * 31 + ch.charCodeAt(0)) % 360;
  return hash;
}

export function shopperName(id) {
  const n = Number(String(id).replace(/\D/g, ""));
  return `Shopper ${n}`;
}

export function shopperIds(count = 24) {
  return Array.from({ length: count }, (_, i) => `user_${String(i * 4 + 1).padStart(5, "0")}`);
}

export function pickRandom(list, except) {
  const pool = list.filter((x) => x !== except);
  return pool[Math.floor(Math.random() * pool.length)];
}

export function clock(date = new Date()) {
  return date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}
