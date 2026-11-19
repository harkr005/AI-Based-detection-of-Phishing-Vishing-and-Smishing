import { userId } from "@/lib/auth";
import { gmailLink, gmailMessageId } from "@/lib/db";
import { fullText } from "@/lib/gmail";

const fail = (error, status = 400) => Response.json({ error }, { status });

// GET ?id=<scan id> -> the full text of that mail, read live from Gmail.
export async function GET(req) {
  const uid = await userId();
  if (!uid) return fail("Sign in first.", 401);

  const scanId = Number(new URL(req.url).searchParams.get("id"));
  if (!Number.isInteger(scanId)) return fail("Missing scan id.");

  try {
    const [messageId, link] = await Promise.all([gmailMessageId(uid, scanId), gmailLink(uid)]);
    if (!messageId) return fail("No such Gmail scan.", 404);
    if (!link?.gmail_token) return fail("Reconnect Gmail to read the full mail.", 409);
    return Response.json(await fullText(link.gmail_token, messageId));
  } catch (e) {
    if (e.status === 404) return fail("This mail is no longer in Gmail.", 404);
    if (e.code === "invalid_grant") return fail("Reconnect Gmail to read the full mail.", 409);
    return fail(e.message, 502);
  }
}
