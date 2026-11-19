import { addressOf, markFeedback, uncontact } from "@/lib/db";
import { userId } from "@/lib/auth";
import { postJSON } from "@/lib/sidecar";

// The user's report on a scan: 'not_a_threat', 'is_spam', or 'is_a_threat'
// (a scam - the name predates the spam/scam split, and old rows carry it). On
// a Gmail scan, 'is_a_threat' alone is a crowd report against the sender
// (store.crowd reads it; it needs behaviour.CROWD_MIN distinct users before it
// moves anybody's score). Spam is unwanted, not fraud, so it never counts.
//
// 'not_a_threat' on a Gmail scan also vouches for the sender: it becomes a
// contact, like someone this mailbox wrote to, so their next mail is not a
// first contact and the behaviour gate stops holding it as SUSPICIOUS. A
// blanket exemption for DMARC-passing top-1M domains was measured and refused
// (scratch/_owner_novelty_eval.py: 5.7% of real phishing qualifies); this one
// only ever covers senders the user named. Calling them a scam or spam later
// takes the vouch back.
const KINDS = new Set(["not_a_threat", "is_spam", "is_a_threat"]);

async function vouch(uid, row, trust) {
  if (trust) {
    // A failed DMARC means the From address is a claim, not the real sender.
    if (String(row.dmarc ?? "").toLowerCase() === "fail") return;
    // ONE address, read the way the scan reads it. The raw From line would go
    // through getaddresses, which can pull a second address out of a crafted
    // display name and vouch for someone the user never saw.
    const addr = addressOf(row.sender);
    if (addr) await postJSON("/email/seed", { user_id: String(uid), contacts: [addr] });
    return;
  }
  await uncontact(uid, row.sender);
  // Evict the sidecar's cached profile so it reloads without the contact.
  await postJSON("/email/forget", { user_id: String(uid) }).catch(() => {});
}

export async function POST(req) {
  const uid = await userId();
  if (!uid) return Response.json({ error: "Unauthorised" }, { status: 401 });

  const { id, feedback = "not_a_threat" } = await req.json();
  if (!id) return Response.json({ error: "No scan to correct" }, { status: 400 });
  if (!KINDS.has(feedback)) return Response.json({ error: "Unknown feedback" }, { status: 400 });

  try {
    const row = await markFeedback(uid, id, feedback);
    if (!row) return Response.json({ error: "Scan not found" }, { status: 404 });
    // Gmail only: a pasted mail's sender is whatever was typed, nothing to vouch for.
    if (row.source === "gmail" && row.sender) await vouch(uid, row, feedback === "not_a_threat");
    return Response.json({ ok: true });
  } catch (e) {
    return Response.json({ error: e.message }, { status: 500 });
  }
}
