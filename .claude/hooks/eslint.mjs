// PostToolUse (Edit|Write): eslint --fix on web/ JS files. Whatever --fix can't
// settle comes back to Claude on stderr (exit 2); the edit itself stands.
import { spawnSync } from "node:child_process";
import { join, relative } from "node:path";

let raw = "";
for await (const chunk of process.stdin) raw += chunk;
const input = JSON.parse(raw);
const file = input.tool_input?.file_path ?? input.tool_response?.filePath ?? "";

const web = join(process.env.CLAUDE_PROJECT_DIR || process.cwd(), "web");
const rel = relative(web, file);
if (rel.startsWith("..") || !/\.(js|jsx|mjs)$/.test(rel)) process.exit(0);

// The repo's own eslint, run by node directly: no npx, no shell quoting.
const out = spawnSync(process.execPath,
  [join(web, "node_modules/eslint/bin/eslint.js"), "--fix", rel],
  { cwd: web, encoding: "utf8" });
if (out.status === 1) {
  console.error(`eslint (web/${rel.replaceAll("\\", "/")}):\n${out.stdout}`);
  process.exit(2);
}
