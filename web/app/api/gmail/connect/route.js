import { userId } from "@/lib/auth";
import { authUrl } from "@/lib/gmail";
import { makeState } from "@/lib/hash";

// Step one of Google's OAuth: send the browser to the consent screen. `state`
// is a ten-minute token naming THIS account, and the callback refuses one that
// names anybody else — otherwise a crafted link could attach an attacker's
// Gmail to a victim's account.
export async function GET(req) {
  const { origin } = new URL(req.url);
  const uid = await userId();
  if (!uid) return Response.redirect(`${origin}/sign-in`);

  const missing = ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "AUTH_SECRET"]
    .find((k) => !process.env[k]);
  if (missing) {
    return Response.redirect(
      `${origin}/gmail?gmail=${encodeURIComponent(`${missing} is missing from .env.local`)}`);
  }

  return Response.redirect(authUrl(`${origin}/api/gmail/callback`,
                                   makeState(process.env.AUTH_SECRET, uid)));
}
