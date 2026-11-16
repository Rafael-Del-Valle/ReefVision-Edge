import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: true,
    globals: true,
    exclude: ["tests/e2e/**", "**/node_modules/**", "**/dist/**"],
  },
  build: {
    outDir: "../src/reefvision_edge/static",
    emptyOutDir: true,
  },
});
