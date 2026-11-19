import {
  expireGmail, gmailLinks, gmailSeeded, hasScan, markGmailSeeded, saveScan,
  setGmailHistory,
} from "./db.js";
import { seal, unseal } from "./hash.js";
import { postJSON } from "./sidecar.js";

// Gmail, POLLED rather than pushed. users.watch needs a Pub/Sub topic, a public
// HTTPS endpoint and a renewal every seven days; polling history.list needs
// none of them, and every step after "something changed" is the same code. The
// price is up to POLL_MS of delay. If that ever matters, watch goes on top of
// this and only the trigger changes.
//
// ponytail: plain fetch against three Google endpoints, not the googleapis
// package — that is a very large dependency for five calls.

export const SCOPE = "https://www.googleapis.com/auth/gmail.readonly";
export const POLL_MS = 60_000;
const API = "https://gmail.googleapis.com/gmail/v1/users/me";

// What gets scanned. messageAdded also fires for the user's own sent mail and
// drafts, which are not threats to them.
const SCANNED = ["INBOX", "SPAM"];

// Mail scanned on connect, and how much history seeds the sender profiles
// before it. SEED_INBOX senders are what arm behaviour.MIN_PROFILE (20) on day
// one; SEED_SENT is who this mailbox writes to, the "have you ever replied"
// signal Gmail leans on.
export const RECENT = 10;
const SEED_INBOX = 200, SEED_SENT = 100;

// Scans from Gmail share the phone's idempotency columns: device 'gmail',
// hash = Gmail's message id, so the partial unique index covers them too.
const DEVICE = "gmail";

function secret() {
  const s = process.env.AUTH_SECRET;
  if (!s) throw new Error("AUTH_SECRET is missing from .env.local");
  return s;
}

export const sealToken = (token) => seal(secret(), token);

/* ---- OAuth ---------------------------------------------------------------- */

export function authUrl(redirectUri, state) {
  return "https://accounts.google.com/o/oauth2/v2/auth?" + new URLSearchParams({
    client_id: process.env.GOOGLE_CLIENT_ID,
    redirect_uri: redirectUri,
    response_type: "code",
    scope: SCOPE,
    // offline = a refresh token; consent = one EVERY time, since Google only
    // issues it on the first grant and a reconnect would otherwise get none.
    access_type: "offline",
    prompt: "consent",
    state,
  });
}

async function tokenCall(params) {
  const res = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    body: new URLSearchParams({
      client_id: process.env.GOOGLE_CLIENT_ID,
      client_secret: process.env.GOOGLE_CLIENT_SECRET,
      ...params,
    }),
  });
  const out = await res.json();
  if (!res.ok) {
    const e = new Error(out.error_description || out.error || `Google ${res.status}`);
    e.code = out.error;
    throw e;
  }
  return out;
}

export const exchangeCode = (code, redirectUri) =>
  tokenCall({ code, redirect_uri: redirectUri, grant_type: "authorization_code" });

// ponytail: a fresh access token every tick rather than caching the hour-long
// one — one extra call per user per minute, and nothing to invalidate.
const accessToken = async (sealed) => {
  const key = secret(); // outside the try: a MISSING secret must not expire anyone
  let refresh;
  try {
    refresh = unseal(key, sealed);
  } catch {
    // AUTH_SECRET was rotated (see hash.js seal): as dead as a revoked grant,
    // so it takes the same path - reconnect - instead of failing every tick.
    throw Object.assign(new Error("stored Gmail token no longer opens"), { code: "invalid_grant" });
  }
  return (await tokenCall({ refresh_token: refresh, grant_type: "refresh_token" })).access_token;
};

// Best-effort: disconnecting must succeed locally even if Google is down.
export async function revoke(sealed) {
  try {
    await fetch("https://oauth2.googleapis.com/revoke", {
      method: "POST",
      body: new URLSearchParams({ token: unseal(secret(), sealed) }),
    });
  } catch {}
}

export async function gmail(access, path) {
  const res = await fetch(`${API}/${path}`, {
    headers: { Authorization: `Bearer ${access}` },
  });
  if (!res.ok) {
    const e = new Error(`Gmail ${res.status}: ${await res.text()}`);
    e.status = res.status;
    throw e;
  }
  return res.json();
}

/* ---- polling -------------------------------------------------------------- */

async function scanMessage(uid, access, id) {
  if (await hasScan(uid, DEVICE, id)) return;

  let m;
  try {
    m = await gmail(access, `messages/${id}?format=raw`);
  } catch (e) {
    // Deleted between history.list and here. Throwing would hold the
    // historyId back and retry a message that no longer exists, forever.
    if (e.status === 404) return;
    throw e;
  }

  let out;
  try {
    out = await postJSON("/email/raw", { raw: m.raw, user_id: String(uid) });
  } catch (e) {
    // Sidecar down: throw, so the historyId is not advanced and this mail is
    // picked up on a later tick. Any other failure belongs to THIS message
    // (no text part, a parser crash) and must not block the ones behind it.
    if (e.message.startsWith("Model server unreachable")) throw e;
    console.warn(`gmail: skipped message ${id}: ${e.message}`);
    return;
  }

  const { preview, sender, ...result } = out;
  // Gmail's own labels ride along in detail: its SPAM call is free ground truth
  // for evaluation later (training on it would be imitating Gmail — say so).
  await saveScan(String(uid), "email", preview, { ...result, gmail_labels: m.labelIds },
                 { source: DEVICE, sender, deviceId: DEVICE, messageHash: id });
}

// Headers only (format=metadata, fields= below): no body text, not even the
// snippet, leaves Gmail on this path. null when
// the mail was deleted in between, like scanMessage's 404.
async function meta(access, id, names) {
  let m;
  try {
    // fields= drops `snippet`, which format=metadata still returns and which
    // is body text: this path must never be able to start reading it.
    m = await gmail(access, `messages/${id}?format=metadata&fields=internalDate,payload/headers&`
                            + names.map((n) => `metadataHeaders=${n}`).join("&"));
  } catch (e) {
    if (e.status === 404) return null;
    throw e;
  }
  const h = Object.fromEntries((m.payload?.headers ?? []).map((x) => [x.name.toLowerCase(), x.value]));
  return { h, at: m.internalDate ? new Date(Number(m.internalDate)).toISOString() : null };
}

const recipients = async (access, ids) => {
  const out = [];
  for (const id of ids) {
    const m = await meta(access, id, ["To", "Cc"]);
    if (m) out.push(...[m.h.to, m.h.cc].filter(Boolean));
  }
  return out;
};

// Once per user, on connect, BEFORE scanRecent: fold the headers of older inbox
// mail and recent Sent mail into the sender profiles, without scoring any of
// it. Skips the RECENT newest inbox mails, which scanRecent is about to score -
// seeding them too would make each one its own history.
// ponytail: sequential metadata calls (~300, under a minute, in after()); batch
// them if connect ever feels slow.
export async function seedProfiles(uid, access, skip = RECENT) {
  if (await gmailSeeded(uid)) return; // a reconnect must not count everyone twice
  const list = async (label, n) =>
    (await gmail(access, `messages?maxResults=${n}&labelIds=${label}`)).messages ?? [];
  const events = [];
  for (const { id } of (await list("INBOX", SEED_INBOX + skip)).slice(skip)) {
    const m = await meta(access, id, ["From"]);
    if (m?.h.from) events.push({ sender: m.h.from, sent_at: m.at });
  }
  const contacts = await recipients(access, (await list("SENT", SEED_SENT)).map((m) => m.id));
  await postJSON("/email/seed", { user_id: String(uid), events, contacts });
  await markGmailSeeded(uid);
}

// Run once right after connecting, so the page is not empty until the next
// mail arrives. INBOX only, and oldest first: scoring also builds the sender
// profiles, and "have we seen this sender before" has to mean before.
export async function scanRecent(uid, access, n = RECENT) {
  const { messages = [] } = await gmail(access, `messages?maxResults=${n}&labelIds=INBOX`);
  for (const { id } of messages.reverse()) await scanMessage(uid, access, id);
}

// The full mail, laid out for reading, fetched live for the View panel. Fetched
// rather than stored: scans keep a 200-character preview, and a whole mailbox
// of bodies at rest is exactly what connecting Gmail should not create.
// Throws with .status 404 when the mail has since been deleted from Gmail.
export async function fullText(sealed, messageId) {
  const access = await accessToken(sealed);
  const m = await gmail(access, `messages/${messageId}?format=raw`);
  return postJSON("/email/parse", { raw: m.raw }); // { subject, sender, text }
}

// Exported for lib/gmail.test.mjs, which must poll ONLY its own user.
export async function pollOne(u) {
  let access;
  try {
    access = await accessToken(u.gmail_token);
  } catch (e) {
    // Revoked by the user, or Testing mode's seven-day refresh-token limit.
    // Either way polling cannot continue until they reconnect.
    if (e.code === "invalid_grant") return expireGmail(u.id);
    throw e;
  }

  const start = u.gmail_history_id;
  let page = "", latest = start;
  const sent = [];
  do {
    let h;
    try {
      h = await gmail(access, `history?historyTypes=messageAdded&startHistoryId=${start}`
                              + (page ? `&pageToken=${page}` : ""));
    } catch (e) {
      // start is older than the history Gmail keeps (about a week, e.g. the
      // server was off). Mail in that gap is not recoverable here; resume now.
      if (e.status !== 404) throw e;
      return setGmailHistory(u.id, (await gmail(access, "profile")).historyId);
    }
    for (const rec of h.history ?? []) {
      for (const { message } of rec.messagesAdded ?? []) {
        if (message.labelIds?.some((l) => SCANNED.includes(l))) {
          await scanMessage(u.id, access, message.id);
        } else if (message.labelIds?.includes("SENT")) {
          sent.push(message.id); // never scanned: only who it went to is kept
        }
      }
    }
    latest = h.historyId;
    page = h.nextPageToken;
  } while (page);

  // Before the historyId moves, so a sidecar outage retries these next tick.
  // Contacts are idempotent, so a retry changes nothing.
  if (sent.length) {
    await postJSON("/email/seed", { user_id: String(u.id),
                                    contacts: await recipients(access, sent) });
  }
  if (latest !== start) await setGmailHistory(u.id, latest);
}

let running = false;

export async function pollAll() {
  // Scoring is slow on a cold sidecar; a tick must not start while the last
  // one is still going, or the same mail is fetched twice in parallel.
  if (running) return;
  running = true;
  try {
    for (const u of await gmailLinks()) {
      try {
        await pollOne(u);
      } catch (e) {
        console.warn(`gmail: user ${u.id}: ${e.message}`);
      }
    }
  } catch (e) {
    console.warn(`gmail: ${e.message}`); // Postgres down; next tick retries
  } finally {
    running = false;
  }
}
