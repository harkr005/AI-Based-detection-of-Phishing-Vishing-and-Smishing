// The Python models live in api.py (FastAPI, port 8000). Everything server-side
// in this app talks to them through here, so there is one place to change when
// the sidecar moves off localhost.
const BASE = process.env.SIDECAR_URL || "http://127.0.0.1:8000";

export async function sidecar(path, init) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    // A dead sidecar is the single most likely failure in dev, and "fetch
    // failed" in a toast tells the user nothing actionable.
    throw new Error(`Model server unreachable at ${BASE} — run: python api.py`);
  }
  if (!res.ok) throw new Error(`Model server ${res.status}: ${await res.text()}`);
  return res.json();
}

export const postJSON = (path, body) =>
  sidecar(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
