import test from "node:test";
import assert from "node:assert/strict";
import { ago, answerShares, lift, serviceState, sparkPath, spendPercent, variantLabel } from "../js/stats.js";

test("serviceState separates up, down and not reported", () => {
  assert.equal(serviceState({ alive: true }), "up");
  assert.equal(serviceState({ alive: false }), "down");
  assert.equal(serviceState(undefined), "unknown");
});
test("ago is short and null-safe", () => {
  assert.equal(ago(null), null);
  assert.equal(ago(0.4), "now");
  assert.equal(ago(12.2), "12 s");
  assert.equal(ago(130), "2 min");
});
test("spendPercent clamps to 0..100", () => {
  assert.equal(spendPercent(0.25, 1), 25);
  assert.equal(spendPercent(5, 1), 100);
  assert.equal(spendPercent(1, 0), 0);
});
test("answerShares always add up to 100 when there are answers", () => {
  const s = answerShares({ ai: 1, template: 1, from_cache: 1 });
  assert.equal(s.ai + s.template + s.from_cache, 100);
  assert.deepEqual(answerShares({}), { ai: 0, template: 0, from_cache: 0, total: 0 });
});
test("lift stays null until both variants have enough lists", () => {
  assert.equal(lift({ control: { lists: 10, rate: 0.2 }, treatment: { lists: 10, rate: 0.3 } }), null);
  assert.equal(lift({ control: { lists: 50, rate: 0.2 }, treatment: { lists: 50, rate: 0.3 } }), 50);
  assert.equal(lift({ control: { lists: 50, rate: 0.2 } }), null);
});
test("sparkPath needs two points and stays inside the box", () => {
  assert.equal(sparkPath([1], 100, 30).line, "");
  const { line, last } = sparkPath([1, 5, 3], 100, 30);
  assert.ok(line.startsWith("M"));
  assert.ok(last[0] <= 100 && last[1] >= 0 && last[1] <= 30);
});
test("variantLabel falls back to the raw name", () => {
  assert.equal(variantLabel("treatment"), "New ranker");
  assert.equal(variantLabel("x"), "x");
});
