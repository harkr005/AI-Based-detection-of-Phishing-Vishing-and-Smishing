import { sidecar } from "@/lib/sidecar";
import { saveScan } from "@/lib/db";
import { userId } from "@/lib/auth";

// Whisper + the spoof head run on the whole recording, so this request is slow
// by construction (~20-60s for a few minutes of audio). The page shows progress
// rather than the request being made async — see the UX note in voice/page.js.
export const maxDuration = 300;

export async function POST(req) {
  const uid = await userId();
  if (!uid) return Response.json({ error: "Unauthorised" }, { status: 401 });

  const form = await req.formData();
  const file = form.get("file");
  if (!file) return Response.json({ error: "No audio uploaded" }, { status: 400 });

  const out = new FormData();
  out.append("file", file, file.name || "call.wav");
  out.append("caller", form.get("caller") || "");
  out.append("user_id", String(uid));

  let result;
  try {
    result = await sidecar("/voice/file", { method: "POST", body: out });
  } catch (e) {
    return Response.json({ error: e.message }, { status: 502 });
  }

  try {
    const row = await saveScan(uid, "voice", file.name || result.transcript || "", result,
                               { sender: String(form.get("caller") || "").slice(0, 200) || null });
    result = { ...result, id: row.id, created_at: row.created_at };
  } catch (e) {
    result = { ...result, saveError: e.message };
  }
  return Response.json(result);
}
