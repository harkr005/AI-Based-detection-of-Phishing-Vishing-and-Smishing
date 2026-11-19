import { postJSON } from "@/lib/sidecar";
import { saveScan } from "@/lib/db";
import { userId } from "@/lib/auth";

// Text-in channels. Voice file upload is multipart and lives in /api/voice.
const ROUTES = {
  // sent_at is the message's own timestamp (OCR reads it off the screenshot),
  // never the moment of the scan — see behaviour.sms_features.
  // first_contact is deliberately NOT defaulted: only the phone's own message
  // store can answer it, and the sidecar abstains on null. user_id lets the
  // sidecar check the sender against this user's profiles anyway.
  sms:   (b, uid) => ["/sms",   { text: b.text, sender: b.sender || "",
                                  first_contact: b.first_contact ?? null,
                                  sent_at: b.sent_at || null,
                                  user_id: String(uid) }],
  // user_id: sender profiles are per mailbox (spec §7) — without it every
  // signed-in user would build one shared profile store. See store.py.
  email: (b, uid) => ["/email", { subject: b.subject || "", body: b.body,
                                  sender: b.sender || "", user_id: String(uid),
                                  sent_at: b.sent_at || null }],
  // caller: the number the call came from, as typed; blank = the caller
  // checks abstain. user_id finds this user's SMS from the same number.
  voice: (b, uid) => ["/voice/text", { text: b.text, caller: b.caller || "",
                                       user_id: String(uid) }],
};

export async function POST(req) {
  const uid = await userId();
  if (!uid) return Response.json({ error: "Unauthorised" }, { status: 401 });

  const body = await req.json();
  const build = ROUTES[body.channel];
  if (!build) return Response.json({ error: "Unknown channel" }, { status: 400 });

  const [path, payload] = build(body, uid);
  if (!(payload.text || payload.body)?.trim())
    return Response.json({ error: "Nothing to scan" }, { status: 400 });

  let result;
  try {
    result = await postJSON(path, payload);
  } catch (e) {
    return Response.json({ error: e.message }, { status: 502 });
  }

  // A scan that cannot be filed is still a valid scan — show it rather than
  // failing the request because Postgres is down.
  try {
    // sender is kept so a later call from the same number can find this SMS
    // (store.risky_sms) - the user's own typed value, on the user's own row.
    // Email's From is not kept: nothing reads it back.
    const sender = body.channel === "email" ? "" : payload.sender || payload.caller || "";
    const row = await saveScan(uid, body.channel, payload.text || payload.body, result,
                               { sender: sender.slice(0, 200) || null });
    result = { ...result, id: row.id, created_at: row.created_at };
  } catch (e) {
    result = { ...result, saveError: e.message };
  }
  return Response.json(result);
}
