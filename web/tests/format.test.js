import test from "node:test";
import assert from "node:assert/strict";
import { formatPrice, shopperIds, shopperName, pickRandom, categoryHue, clock } from "../js/format.js";

test("formatPrice has no decimals", () => assert.equal(formatPrice(1299), "$1,299"));
test("shopperIds are zero padded and unique", () => {
  const ids = shopperIds(5);
  assert.equal(ids[0], "user_00001");
  assert.equal(new Set(ids).size, 5);
});
test("shopperName reads the number", () => assert.equal(shopperName("user_00017"), "Shopper 17"));
test("pickRandom never returns the excluded id", () => {
  for (let i = 0; i < 50; i++) assert.notEqual(pickRandom(["a", "b", "c"], "a"), "a");
});
test("categoryHue is stable and within a colour wheel", () => {
  assert.equal(categoryHue("Books"), categoryHue("Books"));
  assert.ok(categoryHue("Tents") >= 0 && categoryHue("Tents") < 360);
});
test("different categories usually get different hues", () => {
  assert.notEqual(categoryHue("Books"), categoryHue("Laptops"));
});
test("clock formats as HH:MM", () => assert.match(clock(new Date(2026, 0, 1, 9, 5)), /^\d{2}:\d{2}$/));
