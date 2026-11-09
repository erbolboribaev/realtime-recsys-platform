import { en } from "./locales/en.ts";
import type { TextKey } from "./locales/en.ts";
import { uz } from "./locales/uz.ts";

export type { TextKey };
export type Lang = "en" | "uz";

const DICT: Record<Lang, Record<TextKey, string>> = { en, uz };
let lang: Lang = "en";

export const getLang = (): Lang => lang;

export function setLang(next: string): void {
  lang = next === "uz" ? "uz" : "en";
  document.documentElement.lang = lang;
}

export function t(key: TextKey): string {
  return DICT[lang][key] ?? en[key];
}

/** Fills every element that carries a data-i18n attribute. */
export function applyStatic(root: ParentNode = document): void {
  root.querySelectorAll<HTMLElement>("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n as TextKey);
  });
}
