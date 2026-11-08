// Simple stroke icons per category, drawn on a 48px grid.
const ICONS = {
  "Running shoes": '<path d="M5 30c6 0 9-3 11-8l5 4c3 2 7 2 11 3 4 1 7 3 7 6H5z"/><path d="M5 36h38"/>',
  "Sports watches": '<rect x="13" y="12" width="22" height="24" rx="7"/><path d="M18 12l1-7h10l1 7M18 36l1 7h10l1-7M24 19v6l4 3"/>',
  Backpacks: '<rect x="11" y="14" width="26" height="28" rx="8"/><path d="M18 14c0-6 2-9 6-9s6 3 6 9M17 30h14"/>',
  Headphones: '<path d="M8 28V24a16 16 0 0 1 32 0v4"/><rect x="6" y="28" width="9" height="13" rx="3"/><rect x="33" y="28" width="9" height="13" rx="3"/>',
  Laptops: '<rect x="8" y="10" width="32" height="22" rx="3"/><path d="M4 38h40"/>',
  Smartphones: '<rect x="14" y="5" width="20" height="38" rx="5"/><path d="M21 38h6"/>',
  Cameras: '<rect x="5" y="14" width="38" height="26" rx="5"/><circle cx="24" cy="27" r="8"/><path d="M16 14l3-6h10l3 6"/>',
  "Coffee makers": '<path d="M12 8h24v8H12zM14 16v22h20V16M14 38h20M20 38v-8h8v8"/>',
  Cookware: '<path d="M8 22h28v12a6 6 0 0 1-6 6H14a6 6 0 0 1-6-6z"/><path d="M36 26h8M14 22v-4h16v4"/>',
  Books: '<path d="M8 8h14a4 4 0 0 1 4 4v28a4 4 0 0 0-4-4H8zM40 8H26M40 8v28H26"/>',
  "Board games": '<rect x="8" y="8" width="32" height="32" rx="4"/><path d="M8 24h32M24 8v32"/><circle cx="16" cy="16" r="2"/><circle cx="32" cy="32" r="2"/>',
  "Yoga mats": '<path d="M10 14h28v8H10zM10 22v12a6 6 0 0 0 12 0V22"/>',
  Bicycles: '<circle cx="12" cy="31" r="8"/><circle cx="36" cy="31" r="8"/><path d="M12 31l8-16h10l6 16M20 15l6 16h-14M18 10h6"/>',
  Sunglasses: '<path d="M4 20h40M8 20l2 12a5 5 0 0 0 10 0l1-8h6l1 8a5 5 0 0 0 10 0l2-12"/>',
  Jackets: '<path d="M16 8l-9 8 4 22h10V8zM32 8l9 8-4 22H27V8zM24 8v30M16 8c2 4 14 4 16 0"/>',
  "Desk lamps": '<path d="M14 42h20M24 42V26M24 26l-9-10 6-8 12 8z"/>',
  Keyboards: '<rect x="4" y="14" width="40" height="22" rx="4"/><path d="M10 21h2M16 21h2M22 21h2M28 21h2M34 21h2M12 28h24"/>',
  Monitors: '<rect x="5" y="9" width="38" height="24" rx="3"/><path d="M18 42h12M24 33v9"/>',
  "Water bottles": '<path d="M19 5h10v6l4 5v24a3 3 0 0 1-3 3H18a3 3 0 0 1-3-3V16l4-5zM15 24h18"/>',
  Tents: '<path d="M4 40L24 8l20 32zM24 8v32M24 40l-8-12"/>'
};
const FALLBACK = '<rect x="9" y="9" width="30" height="30" rx="8"/>';
export function iconSvg(category) {
  return `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[category] ?? FALLBACK}</svg>`;
}
