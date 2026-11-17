import path from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixtures = path.resolve(here, "../../.e2e-fixtures");
const screenshots = path.resolve(here, "../../../docs/screenshots");

async function stabilizeScreenshotText(page: import("@playwright/test").Page) {
  await page.evaluate(() => {
    const previewTitle = document.querySelector(".preview-panel h3");
    if (previewTitle) previewTitle.textContent = "Session sample01";
    const selectedSession =
      document.querySelector<HTMLSelectElement>("#active-session");
    if (selectedSession?.selectedOptions[0]) {
      selectedSession.selectedOptions[0].textContent = "sample01 · completed";
    }
    const values = ["8.0 FPS", "12.5 ms", "3", "6"];
    document.querySelectorAll(".metric strong").forEach((element, index) => {
      if (values[index]) element.textContent = values[index];
    });
    document.querySelectorAll("time").forEach((element) => {
      element.textContent = "11/16/2026, 12:00:00 PM";
    });
    document.querySelectorAll(".session-stats strong").forEach((element) => {
      element.textContent = "12.5";
    });
  });
}

test.describe.serial("ReefVision-Edge field workflow", () => {
  test("registers local inputs, runs inference, reloads, and exports", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Field monitor" }),
    ).toBeVisible();
    await page.getByRole("tab", { name: "Models" }).click();
    await page
      .getByLabel("ONNX model")
      .setInputFiles(path.join(fixtures, "tiny.onnx"));
    await page
      .getByLabel("Model metadata")
      .setInputFiles(path.join(fixtures, "metadata.json"));
    await page.getByRole("button", { name: "Register ONNX" }).click();
    await expect(page.getByText("Tiny Coral Detector")).toBeVisible();

    await page.getByRole("tab", { name: "Monitor" }).click();
    await expect(page.getByLabel("Model")).toContainText("Tiny Coral Detector");
    await page.getByText("Register a source").click();

    await page
      .getByLabel("Video file")
      .setInputFiles(path.join(fixtures, "reef.avi"));
    await page.getByRole("button", { name: "Import video" }).click();
    await expect(page.getByLabel("Source")).toContainText("reef.avi");

    await page.getByLabel("Sample every N frames").fill("2");
    await page.getByRole("button", { name: "Start inference" }).click();
    await expect(page.getByRole("status", { name: "completed" })).toBeVisible({
      timeout: 10_000,
    });
    const metrics = page.getByLabel("Session metrics");
    await expect(metrics).toContainText("3");
    await expect(metrics).toContainText("6");
    await expect(
      page.getByRole("img", {
        name: "Latest inference frame with detection overlays",
      }),
    ).toBeVisible();

    await page.reload();
    await expect(page.getByRole("status", { name: "completed" })).toBeVisible();
    await page.getByRole("tab", { name: "Sessions" }).click();
    await expect(page.getByText("6 detections")).toBeVisible();
    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("link", { name: "Export" }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/^reefvision-edge-.*\.zip$/);
  });

  test("matches ReefVision semantic colors in computed styles", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Field monitor" }),
    ).toBeVisible();

    const tokens = await page.evaluate(() => {
      const style = getComputedStyle(document.documentElement);
      return Object.fromEntries(
        [
          "--rv-bg",
          "--rv-bg-raised",
          "--rv-panel",
          "--rv-panel-soft",
          "--rv-border",
          "--rv-border-strong",
          "--rv-accent",
          "--rv-accent-strong",
          "--rv-text",
          "--rv-muted",
          "--rv-muted-strong",
          "--rv-success",
          "--rv-warning",
          "--rv-danger",
          "--rv-info",
        ].map((name) => [name, style.getPropertyValue(name).trim()]),
      );
    });
    const normalizedTokens = Object.fromEntries(
      Object.entries(tokens).map(([name, value]) => [
        name,
        String(value).replace(/, \.(\d+)/g, ", 0.$1"),
      ]),
    );
    expect(normalizedTokens).toEqual({
      "--rv-bg": "#06171c",
      "--rv-bg-raised": "#0a2228",
      "--rv-panel": "rgba(7, 34, 40, 0.94)",
      "--rv-panel-soft": "rgba(12, 47, 54, 0.72)",
      "--rv-border": "rgba(182, 238, 228, 0.14)",
      "--rv-border-strong": "rgba(114, 219, 201, 0.42)",
      "--rv-accent": "#72dbc9",
      "--rv-accent-strong": "#80ddce",
      "--rv-text": "#eaf8f5",
      "--rv-muted": "#9dbdb8",
      "--rv-muted-strong": "#bad7d2",
      "--rv-success": "#72dbc9",
      "--rv-warning": "#f3c96b",
      "--rv-danger": "#ff8f87",
      "--rv-info": "#8fc7ff",
    });
    await expect(page.locator("body")).toHaveCSS("color", "rgb(234, 248, 245)");
    await expect(page.getByRole("tab", { name: "Monitor" })).toHaveCSS(
      "background-color",
      "rgb(114, 219, 201)",
    );
    await expect(page.locator(".preview-panel")).toHaveCSS(
      "border-color",
      "rgba(182, 238, 228, 0.14)",
    );
    const completed = page.getByRole("status", { name: "completed" });
    await expect(completed).toHaveCSS("color", "rgb(114, 219, 201)");
    const primary = page.getByRole("button", { name: "Start inference" });
    await expect(primary).toHaveCSS(
      "background-image",
      "linear-gradient(135deg, rgb(128, 221, 206), rgb(114, 219, 201))",
    );
    const monitorTab = page.getByRole("tab", { name: "Monitor" });
    await monitorTab.focus();
    await expect(monitorTab).toHaveCSS("outline-color", "rgb(114, 219, 201)");
    await completed.evaluate((element) => {
      for (const state of ["running", "interrupted", "failed"]) {
        const clone = element.cloneNode(true) as HTMLElement;
        clone.dataset.state = state;
        clone.setAttribute("aria-label", state);
        clone.id = `theme-${state}`;
        document.body.append(clone);
      }
    });
    await expect(page.locator("#theme-running")).toHaveCSS(
      "color",
      "rgb(143, 199, 255)",
    );
    await expect(page.locator("#theme-interrupted")).toHaveCSS(
      "color",
      "rgb(243, 201, 107)",
    );
    await expect(page.locator("#theme-failed")).toHaveCSS(
      "color",
      "rgb(255, 143, 135)",
    );
  });

  test("captures laptop and compact views", async ({ page }) => {
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Field monitor" }),
    ).toBeVisible();
    await stabilizeScreenshotText(page);
    await page.screenshot({
      path: path.join(screenshots, "monitor-laptop.png"),
      fullPage: true,
    });
    await page.getByRole("tab", { name: "Sessions" }).click();
    await stabilizeScreenshotText(page);
    await page.screenshot({
      path: path.join(screenshots, "sessions-laptop.png"),
      fullPage: true,
    });
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.screenshot({
      path: path.join(screenshots, "settings-laptop.png"),
      fullPage: true,
    });

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("tab", { name: "Monitor" }).click();
    await stabilizeScreenshotText(page);
    await page.screenshot({
      path: path.join(screenshots, "monitor-compact.png"),
      fullPage: true,
    });
    await page.getByRole("tab", { name: "Sessions" }).click();
    await stabilizeScreenshotText(page);
    await page.screenshot({
      path: path.join(screenshots, "sessions-compact.png"),
      fullPage: true,
    });
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.screenshot({
      path: path.join(screenshots, "settings-compact.png"),
      fullPage: true,
    });
  });
});
