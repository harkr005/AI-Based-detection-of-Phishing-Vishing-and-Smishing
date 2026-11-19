import { userId } from "@/lib/auth";
import { deleteGmailScans, forgetContacts, gmailLink, unlinkGmail } from "@/lib/db";
import { revoke } from "@/lib/gmail";
import { postJSON } from "@/lib/sidecar";

const fail = (error, status = 400) => Response.json({ error }, { status });

// What the Gmail card shows. Never the token, sealed or not.
export async function GET() {
  const uid = await userId();
  if (!uid) return fail("Sign in first.", 401);
  try {
    const row = await gmailLink(uid);
    return Response.json({ address: row?.gmail_address ?? null,
                           connected: Boolean(row?.gmail_token) });
  } catch (e) {
    return fail(`Could not reach the database: ${e.message}`, 502);
  }
}

// Disconnect: revoke at Google as well as forgetting it here, so "disconnect"
// means Google stops honouring the grant, not just that we stop asking - and
// the scanned mail goes too. The card warns before calling this.
export async function DELETE() {
  const uid = await userId();
  if (!uid) return fail("Sign in first.", 401);
  try {
    const row = await gmailLink(uid);
    if (row?.gmail_token) await revoke(row.gmail_token);
    await unlinkGmail(uid);
    const deleted = await deleteGmailScans(uid);
    await forgetContacts(uid);
    // Best-effort: the rows are already gone; this only stops the sidecar's
    // in-memory copy treating them as contacts until it restarts.
    await postJSON("/email/forget", { user_id: String(uid) }).catch(() => {});
    return Response.json({ ok: true, deleted });
  } catch (e) {
    return fail(`Could not reach the database: ${e.message}`, 502);
  }
}
