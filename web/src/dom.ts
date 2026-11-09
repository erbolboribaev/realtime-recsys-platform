// A tiny element builder. Text always goes through textContent, so nothing that comes from the
// API is ever parsed as HTML.

type Child = Node | string | null | undefined | Child[];
type Props = Record<string, string | ((event: Event) => void)>;

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, props: Props = {}, ...kids: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "style") el.style.cssText = value;
    else el.setAttribute(key, value);
  }
  el.append(...flatten(kids));
  return el;
}

function flatten(kids: Child[]): (Node | string)[] {
  return kids.flatMap((kid) => (Array.isArray(kid) ? flatten(kid) : kid == null ? [] : [kid]));
}

/** getElementById that fails loudly: a missing element is a bug in the page, not a case to skip. */
export function $<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id} in the page`);
  return el as T;
}
