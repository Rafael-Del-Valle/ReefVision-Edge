import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig, devices } from "@playwright/test";

const webRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:8765",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command:
      '"..\\.venv\\Scripts\\python.exe" -m uvicorn reefvision_edge.app:create_app --factory --host 127.0.0.1 --port 8765',
    cwd: webRoot,
    url: "http://127.0.0.1:8765/api/health",
    reuseExistingServer: false,
    timeout: 30_000,
    env: {
      ...process.env,
      REEFVISION_EDGE_DATA_DIR: path.join(webRoot, "test-results", "e2e-data"),
    },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } },
    },
  ],
});
