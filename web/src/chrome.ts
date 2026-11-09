// Language and theme switches shared by every page.
import { applyStatic, setLang } from "./i18n.ts";

function safe<T>(fn: () => T): T | null {
  try {
    return fn();
  } catch {
    return null;
  }
}

export function initChrome({ onLang = () => {} }: { onLang?: () => void } = {}): void {
  const root = document.documentElement;
  const langButtons = document.querySelectorAll<HTMLButtonElement>("[data-lang]");
  langButtons.forEach((button) =>
    button.addEventListener("click", () => {
      const lang = button.dataset.lang ?? "en";
      setLang(lang);
      safe(() => localStorage.setItem("lang", lang));
      langButtons.forEach((x) => x.setAttribute("aria-pressed", String(x === button)));
      applyStatic();
      onLang();
    }),
  );
  document.getElementById("theme")?.addEventListener("click", () => {
    const dark = root.dataset.theme === "dark" || (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    root.dataset.theme = next;
    safe(() => localStorage.setItem("theme", next));
  });
  const theme = safe(() => localStorage.getItem("theme"));
  if (theme) root.dataset.theme = theme;
  if (safe(() => localStorage.getItem("lang")) === "uz") document.querySelector<HTMLButtonElement>('[data-lang="uz"]')?.click();
}
