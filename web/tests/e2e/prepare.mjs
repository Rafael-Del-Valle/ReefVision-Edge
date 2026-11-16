import { spawnSync } from "node:child_process";
import { rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(here, "../..");
const repoRoot = path.resolve(webRoot, "..");
const dataDir = path.join(webRoot, "test-results", "e2e-data");
const fixtureDir = path.join(webRoot, ".e2e-fixtures");

if (path.dirname(dataDir) !== path.join(webRoot, "test-results")) {
  throw new Error(`Refusing to replace unexpected data path: ${dataDir}`);
}
if (path.dirname(fixtureDir) !== webRoot || path.basename(fixtureDir) !== ".e2e-fixtures") {
  throw new Error(`Refusing to replace unexpected fixture path: ${fixtureDir}`);
}
rmSync(dataDir, { recursive: true, force: true });
rmSync(fixtureDir, { recursive: true, force: true });

const python = path.join(repoRoot, ".venv", "Scripts", "python.exe");
const result = spawnSync(
  python,
  ["-m", "tests.fixtures.prepare_e2e", fixtureDir],
  { cwd: repoRoot, encoding: "utf8" },
);
if (result.status !== 0) {
  throw new Error(`Fixture generation failed:\n${result.stdout}\n${result.stderr}`);
}
