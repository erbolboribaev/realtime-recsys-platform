import test from "node:test";
import assert from "node:assert/strict";
import { en } from "../src/locales/en.ts";
import { uz } from "../src/locales/uz.ts";

test("both languages have the same keys (the compiler checks this too)", () => {
  assert.deepEqual(Object.keys(uz).sort(), Object.keys(en).sort());
});

test("no text is empty", () => {
  for (const [key, value] of [...Object.entries(en), ...Object.entries(uz)]) assert.ok(value.trim().length > 0, key);
});

test("every key used in the HTML pages exists", async () => {
  const { readFileSync } = await import("node:fs");
  for (const page of ["index.html", "admin.html"]) {
    const html = readFileSync(new URL(`../${page}`, import.meta.url), "utf8");
    for (const [, key] of html.matchAll(/data-i18n="([^"]+)"/g)) assert.ok(key! in en, `${page}: ${key}`);
  }
});
