import { after } from "next/server";

import { userId } from "@/lib/auth";
import { linkGmail } from "@/lib/db";
import { SCOPE, exchangeCode, gmail, scanRecent, sealToken, seedProfiles } from "@/lib/gmail";
import { readState } from "@/lib/hash";

// Step two: Google sends the browser back here with a one-time code. Every
// outcome returns to the /gmail page with ?gmail=<what happened>, which the
// Gmail card reads.
export async function GET(req) {
  const url = new URL(req.url);
  const back = (msg) =>
    Response.redirect(`${url.origin}/gmail?gmail=${encodeURIComponent(msg)}`);

  const uid = await userId();
  if (!uid || readState(process.env.AUTH_SECRET, url.searchParams.get("state")) !== String(uid)) {
    return back("That Gmail sign-in did not match this session. Try again.");
  }
  if (!url.searchParams.get("code")) return back("Gmail was not connected.");

  try {
    const t = await exchangeCode(url.searchParams.get("code"), `${url.origin}/api/gmail/callback`);
    // Google's consent screen lets the user untick the scope and still press
    // Continue. Without it every poll would 403, so say so now.
    if (!t.scope?.split(" ").includes(SCOPE)) {
      return back("Gmail was not connected: the 'read your email' permission was left unticked.");
    }
    if (!t.refresh_token) return back("Google did not issue a refresh token. Try again.");

    // Polling starts from NOW; the last ten inbox mails are scanned once so
    // the page has something on it. After the redirect, not before it: ten
    // mails through a cold sidecar is tens of seconds of a blank browser tab.
    const p = await gmail(t.access_token, "profile");
    await linkGmail(uid, p.emailAddress.toLowerCase(), sealToken(t.refresh_token), p.historyId);
    // Seed the sender profiles from mailbox headers first, so those ten are
    // scored against a real history. A failed seed still lets the scan run.
    after(() => seedProfiles(uid, t.access_token)
      .catch((e) => console.warn(`gmail: seeding for user ${uid}: ${e.message}`))
      .then(() => scanRecent(uid, t.access_token))
      .catch((e) => console.warn(`gmail: first scan for user ${uid}: ${e.message}`)));
    return back("connected");
  } catch (e) {
    if (e.code === "23505") return back("That Gmail is already connected to another account.");
    return back(`Gmail was not connected: ${e.message}`);
  }
}
