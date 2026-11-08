// Pure helpers for the dashboard, covered by node tests.

export const MIN_LISTS_FOR_COMPARISON = 30;
const VARIANT_LABELS = { control: "Baseline order", treatment: "New ranker", cold_start: "New shoppers", popularity: "Popularity", ranker_v2: "New ranker" };

export function variantLabel(name) {
  return VARIANT_LABELS[name] ?? name;
}

/** "up" when the service reported recently, "down" when it did not, "unknown" when it is not reported at all. */
export function serviceState(service) {
  if (!service) return "unknown";
  return service.alive ? "up" : "down";
}

export function ago(seconds) {
  if (seconds == null) return null;
  if (seconds < 2) return "now";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  return `${Math.round(seconds / 60)} min`;
}

export function spendPercent(spent, budget) {
  if (!budget || budget <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((spent / budget) * 100)));
}

/** Shares of the three answer kinds, as whole percents that add up to 100 (or all 0 with no answers). */
export function answerShares(answers) {
  const keys = ["ai", "template", "from_cache"];
  const total = keys.reduce((sum, k) => sum + (answers?.[k] ?? 0), 0);
  if (!total) return { ai: 0, template: 0, from_cache: 0, total: 0 };
  const shares = {};
  let used = 0;
  keys.forEach((k, i) => {
    shares[k] = i === keys.length - 1 ? 100 - used : Math.round(((answers[k] ?? 0) / total) * 100);
    used += shares[k];
  });
  return { ...shares, total };
}

/**
 * Relative difference of the new ranker's action rate over the baseline's.
 * Returns null until both sides have enough lists, so a handful of clicks never reads as a result.
 */
export function lift(variants, minLists = MIN_LISTS_FOR_COMPARISON) {
  const a = variants?.control;
  const b = variants?.treatment;
  if (!a || !b || a.lists < minLists || b.lists < minLists || !a.rate) return null;
  return Math.round(((b.rate - a.rate) / a.rate) * 1000) / 10;
}

/** SVG path for a sparkline in a width x height box; returns line and closed area. */
export function sparkPath(values, width, height, pad = 3) {
  const nums = values.filter((v) => typeof v === "number");
  if (nums.length < 2) return { line: "", area: "", last: null };
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  const span = max - min || 1;
  const step = (width - pad * 2) / (nums.length - 1);
  const points = nums.map((v, i) => [pad + i * step, height - pad - ((v - min) / span) * (height - pad * 2)]);
  const line = points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("");
  const [lx, ly] = points[points.length - 1];
  return { line, area: `${line}L${lx.toFixed(1)} ${height}L${pad} ${height}Z`, last: [lx, ly] };
}
