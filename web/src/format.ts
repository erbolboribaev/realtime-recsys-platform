// Pure helpers, covered by tests.

export function formatPrice(value: number): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);
}

/** A stable hue (0-359) per category, so every category keeps its own colour. */
export function categoryHue(category: string): number {
  let hash = 0;
  for (const ch of String(category)) hash = (hash * 31 + ch.charCodeAt(0)) % 360;
  return hash;
}

export function shopperName(id: string): string {
  return `Shopper ${Number(String(id).replace(/\D/g, ""))}`;
}

export function shopperIds(count = 24): string[] {
  return Array.from({ length: count }, (_, i) => `user_${String(i * 4 + 1).padStart(5, "0")}`);
}

export function pickRandom<T>(list: readonly T[], except?: T): T {
  const pool = list.filter((x) => x !== except);
  return pool[Math.floor(Math.random() * pool.length)] as T;
}

export function clock(date: Date = new Date()): string {
  return date.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}
