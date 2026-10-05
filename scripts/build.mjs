// Compiles src/*.ts to dist/*.js for Node, which does not run TypeScript from node_modules (Bun runs
// the sources directly). Node's own type stripping, plus "./x.ts" imports rewritten to "./x.js".
// Runs on install from git (the "prepare" script) and needs no dependencies.
import fs from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const src = path.join(root, "src"), out = path.join(root, "dist");
fs.rmSync(out, { recursive: true, force: true });
for (const file of fs.readdirSync(src, { recursive: true })) {
  if (!file.endsWith(".ts") || file.endsWith(".test.ts")) continue;
  const code = stripTypeScriptTypes(fs.readFileSync(path.join(src, file), "utf8"));
  const js = code.replace(/((?:from|import)\s*\(?\s*["'])(\.{1,2}\/[^"']+)\.ts(["'])/g, "$1$2.js$3");
  const target = path.join(out, file.replace(/\.ts$/, ".js"));
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, js);
}
console.log(`Built ${path.relative(root, out)}/`);
