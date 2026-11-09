import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// In development the page runs on Vite's server and the API calls go to the backend on :8000.
const backend = process.env.API_URL ?? "http://127.0.0.1:8000";
const routes = ["/recommend", "/items", "/events", "/explain", "/admin/stats", "/health"];

export default defineConfig({
  build: {
    outDir: "dist",
    rollupOptions: {
      input: {
        index: fileURLToPath(new URL("./index.html", import.meta.url)),
        admin: fileURLToPath(new URL("./admin.html", import.meta.url)),
      },
    },
  },
  server: {
    proxy: Object.fromEntries(routes.map((route) => [route, backend])),
  },
});
