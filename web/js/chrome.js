// Language and theme switches shared by every page.
import { applyStatic, setLang } from "./i18n.js";

function safe(fn) { try { return fn(); } catch { return null; } }

export function initChrome({ onLang = () => {} } = {}) {
  const root = document.documentElement;
  document.querySelectorAll("[data-lang]").forEach((button) => button.addEventListener("click", () => {
    setLang(button.dataset.lang);
    safe(() => localStorage.setItem("lang", button.dataset.lang));
    document.querySelectorAll("[data-lang]").forEach((x) => x.setAttribute("aria-pressed", String(x === button)));
    applyStatic();
    onLang();
  }));
  document.getElementById("theme").addEventListener("click", () => {
    const dark = root.dataset.theme === "dark" || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    root.dataset.theme = next;
    safe(() => localStorage.setItem("theme", next));
  });
  const theme = safe(() => localStorage.getItem("theme"));
  if (theme) root.dataset.theme = theme;
  if (safe(() => localStorage.getItem("lang")) === "uz") document.querySelector('[data-lang="uz"]').click();
}
