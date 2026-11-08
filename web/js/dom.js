// Tiny element builder. Text always goes through textContent, so nothing from the API is parsed as HTML.
export function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "style") el.style.cssText = value;
    else el.setAttribute(key, value);
  }
  kids.flat().forEach((kid) => kid != null && el.append(kid));
  return el;
}

export const $ = (id) => document.getElementById(id);
