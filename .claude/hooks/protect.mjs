// PreToolUse (Edit|Write|NotebookEdit): refuse hand edits to secrets and to
// trained artifacts. Exit 2 blocks the tool call and shows stderr to Claude.
//
//   .env*                 AUTH_SECRET lives here; changing it signs everyone out
//                         AND disconnects every Gmail (lib/hash.js seal).
//   models/               checkpoints; they change by retraining, never by hand.
//   android assets *.bin  exported weights; they change via export_android.py.
let raw = "";
for await (const chunk of process.stdin) raw += chunk;
const path = String(JSON.parse(raw).tool_input?.file_path ?? "").replaceAll("\\", "/");

const name = path.split("/").pop();
const reason =
  /^\.env(\.|$)/.test(name) && !name.endsWith(".example") ? "a secrets file (.env*)"
  : /\/models\//.test(path) ? "a trained model under models/ - retrain instead"
  : /\/android\/app\/src\/main\/assets\/.*\.bin$/.test(path)
    ? "exported weights - run `python export_android.py` instead"
  : null;

if (reason) {
  console.error(`Blocked: ${path} is ${reason}. Ask the user to change it themselves.`);
  process.exit(2);
}
