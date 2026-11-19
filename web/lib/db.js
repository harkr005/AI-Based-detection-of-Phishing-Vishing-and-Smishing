import { Pool } from "pg";

// Two tables here: `users` holds the login (password never stored in the clear
// — see lib/auth.js) and `scans` is the history the dashboard cards and feeds
// read off. A third, `sender_profiles`, lives in the same database but is owned
// by store.py — Python is its only writer, and api.py has to work whether or not
// this app was ever started.
const SCHEMA = `
CREATE TABLE IF NOT EXISTS users (
  id            BIGSERIAL PRIMARY KEY,
  username      TEXT        NOT NULL UNIQUE,
  password_hash TEXT        NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS scans (
  id          BIGSERIAL PRIMARY KEY,
  user_id     TEXT        NOT NULL,
  channel     TEXT        NOT NULL,
  preview     TEXT        NOT NULL,
  verdict     TEXT        NOT NULL,
  risk        REAL        NOT NULL,
  reason      TEXT,
  detail      JSONB       NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS scans_user_time ON scans (user_id, created_at DESC);

-- "This was not a threat", from the results panel. A column rather than a table:
-- it is a correction to a row that already exists. IF NOT EXISTS so an existing
-- database picks it up on the next query.
ALTER TABLE scans ADD COLUMN IF NOT EXISTS feedback TEXT;
-- When it was given: crowd reports (store.crowd) only count for 90 days.
-- 'is_a_threat' on a Gmail scan is a crowd report against its sender.
ALTER TABLE scans ADD COLUMN IF NOT EXISTS feedback_at TIMESTAMPTZ;

-- PROJECT_SPEC.md §10: both deployment modes emit an identical event schema
-- with a source field, so one table and one feed serve the dashboard and the
-- phone alike. 'web' as the default is what every row written before the phone
-- existed actually was. (No backticks in here: this is a JS template literal.)
ALTER TABLE scans ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'web';

-- Who sent it: the phone's sender, or the number typed with a pasted SMS or a
-- voice call (store.risky_sms matches calls to SMS on it). Nullable rather than
-- defaulted to a lie, since a paste often has no sender.
ALTER TABLE scans ADD COLUMN IF NOT EXISTS sender TEXT;

-- The Gmail feed filters on source, which the user_id+time index above
-- cannot serve without a filter step.
CREATE INDEX IF NOT EXISTS scans_user_source_time
  ON scans (user_id, source, created_at DESC);

-- Idempotency for ingested mail: lib/gmail.js files each message once
-- (device_id 'gmail', message_hash = Gmail's message id). Partial, so the
-- website's own scans (which have neither column) are exempt rather than
-- relying on NULLs comparing unequal.
ALTER TABLE scans ADD COLUMN IF NOT EXISTS device_id    TEXT;
ALTER TABLE scans ADD COLUMN IF NOT EXISTS message_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS scans_device_message
  ON scans (user_id, device_id, message_hash)
  WHERE device_id IS NOT NULL AND message_hash IS NOT NULL;

-- Gmail, polled by lib/gmail.js. Columns on users rather than a table of their
-- own: one mailbox per account, so a table would only repeat the key. The token
-- is sealed (lib/hash.js). A NULL token beside a kept address means Google
-- stopped honouring it — the card asks for a reconnect instead of going blank.
ALTER TABLE users ADD COLUMN IF NOT EXISTS gmail_address    TEXT UNIQUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS gmail_token      TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS gmail_history_id TEXT;
-- Set once lib/gmail.js seedProfiles has folded this mailbox's history into the
-- sender profiles. Kept across disconnects, like the profiles themselves, so a
-- reconnect does not count every sender twice.
ALTER TABLE users ADD COLUMN IF NOT EXISTS gmail_seeded_at  TIMESTAMPTZ;
`;

// Next dev reloads this module on every edit; a fresh Pool each time leaks
// connections until Postgres refuses new ones.
function pool() {
  if (!globalThis._pgPool) {
    globalThis._pgPool = new Pool({ connectionString: process.env.DATABASE_URL });
    globalThis._pgReady = globalThis._pgPool.query(SCHEMA);
  }
  return globalThis._pgPool;
}

async function q(text, params) {
  const p = pool();
  await globalThis._pgReady;
  return p.query(text, params);
}

/* ---- users ------------------------------------------------------------- */

// Usernames are compared case-insensitively so "Mohak" and "mohak" cannot both
// exist; the UNIQUE index is on the stored (already lowered) value.
export async function createUser(username, passwordHash) {
  const { rows } = await q(
    `INSERT INTO users (username, password_hash) VALUES ($1, $2)
     ON CONFLICT (username) DO NOTHING
     RETURNING id, username`,
    [username.toLowerCase(), passwordHash]);
  return rows[0] ?? null; // null = the name is taken
}

export async function findUser(username) {
  const { rows } = await q(
    `SELECT id, username, password_hash FROM users WHERE username = $1`,
    [username.toLowerCase()]);
  return rows[0] ?? null;
}

export async function findUserById(id) {
  const { rows } = await q(`SELECT id, username FROM users WHERE id = $1`, [id]);
  return rows[0] ?? null;
}

/* ---- gmail --------------------------------------------------------------- */

export async function linkGmail(userId, address, sealedToken, historyId) {
  await q(
    `UPDATE users SET gmail_address = $2, gmail_token = $3, gmail_history_id = $4
      WHERE id = $1`,
    [userId, address, sealedToken, String(historyId)]);
}

// Server-side only: the sealed token must never be sent to a browser.
export async function gmailLink(userId) {
  const { rows } = await q(
    `SELECT gmail_address, gmail_token FROM users WHERE id = $1`, [userId]);
  return rows[0] ?? null;
}

export async function unlinkGmail(userId) {
  await q(
    `UPDATE users SET gmail_address = NULL, gmail_token = NULL,
                      gmail_history_id = NULL WHERE id = $1`, [userId]);
}

// Disconnecting takes the scanned mail with it. Sender profiles (store.py)
// stay: they are per mailbox, shared with pasted mail, and hold no text.
export async function deleteGmailScans(userId) {
  const { rowCount } = await q(
    `DELETE FROM scans WHERE user_id = $1 AND source = 'gmail'`, [String(userId)]);
  return rowCount;
}

// Disconnecting forgets who this mailbox WROTE to (lib/gmail.js seedProfiles
// and the Sent poll): rows that exist only as contacts go, and the flag comes
// off the rest. The one write this app makes to store.py's table - here rather
// than through the sidecar so a dead sidecar cannot block a deletion. The
// sidecar's cached copy is evicted separately (/email/forget). Absent table
// (sidecar never run) = nothing to forget.
export async function forgetContacts(userId) {
  try {
    const { rowCount } = await q(
      `DELETE FROM sender_profiles WHERE user_id = $1 AND contacted AND n = 0`,
      [String(userId)]);
    await q(`UPDATE sender_profiles SET contacted = false WHERE user_id = $1 AND contacted`,
            [String(userId)]);
    return rowCount;
  } catch (e) {
    if (e.code === "42P01" || e.code === "42703") return 0; // no table / column yet
    throw e;
  }
}

// What the poller walks every tick.
export async function gmailLinks() {
  const { rows } = await q(
    `SELECT id, gmail_token, gmail_history_id FROM users WHERE gmail_token IS NOT NULL`);
  return rows;
}

export async function gmailSeeded(userId) {
  const { rows } = await q(`SELECT gmail_seeded_at FROM users WHERE id = $1`, [userId]);
  return Boolean(rows[0]?.gmail_seeded_at);
}

export async function markGmailSeeded(userId) {
  await q(`UPDATE users SET gmail_seeded_at = now() WHERE id = $1`, [userId]);
}

export async function setGmailHistory(userId, historyId) {
  await q(`UPDATE users SET gmail_history_id = $2 WHERE id = $1`,
          [userId, String(historyId)]);
}

// Keeps the address, so the card can say "reconnect" rather than go blank.
export async function expireGmail(userId) {
  await q(`UPDATE users SET gmail_token = NULL WHERE id = $1`, [userId]);
}

// Which Gmail message a scan row was. Scoped by user_id: the scan id comes from
// the browser, and without it anyone could read anyone's mail through View.
// One Gmail scan for its own page (/gmail/[id]), same columns as feedScans.
export async function gmailScan(userId, scanId) {
  const { rows } = await q(
    `SELECT id, sender, preview, verdict, risk, reason, detail, created_at
       FROM scans WHERE id = $1 AND user_id = $2 AND source = 'gmail'`,
    [scanId, String(userId)]);
  return rows[0] ?? null;
}

export async function gmailMessageId(userId, scanId) {
  const { rows } = await q(
    `SELECT message_hash FROM scans
      WHERE id = $1 AND user_id = $2 AND source = 'gmail'`,
    [scanId, String(userId)]);
  return rows[0]?.message_hash ?? null;
}

// Asked BEFORE scoring, not left to saveScan's ON CONFLICT: scoring a message
// also files its sender into the behaviour profile, so a retried batch would
// count every sender twice even though the scan row itself deduped.
export async function hasScan(userId, deviceId, messageHash) {
  const { rows } = await q(
    `SELECT 1 FROM scans WHERE user_id = $1 AND device_id = $2 AND message_hash = $3`,
    [String(userId), deviceId, messageHash]);
  return rows.length > 0;
}

/* ---- scans ------------------------------------------------------------- */

export async function saveScan(userId, channel, preview, result, opts = {}) {
  // ponytail: `preview` is capped at 200 chars on purpose — §10's Preview
  // privacy mode. Storing full bodies is the Contribute mode and needs consent,
  // so it is deliberately not what this writes.
  const { source = "web", sender = null, deviceId = null, messageHash = null } = opts;
  // ON CONFLICT DO NOTHING makes a replayed upload a no-op instead of a second
  // row; null back means "already had it", which is a success, not a failure.
  // Web scans pass neither id nor hash, fall outside the partial index, and so
  // can never conflict.
  const { rows } = await q(
    `INSERT INTO scans
       (user_id, channel, preview, verdict, risk, reason, detail, source, sender,
        device_id, message_hash)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
     ON CONFLICT DO NOTHING
     RETURNING id, created_at`,
    [userId, channel, (preview || "").slice(0, 200), result.verdict,
     result.risk, result.reason || null, result, source, sender,
     deviceId, messageHash]);
  return rows[0] ?? null;
}

/* ---- feeds ------------------------------------------------------------- */

// One source's own page ('gmail' for the inbox poller). `detail` rides along:
// the View panel renders straight from it.
export async function feedScans(userId, source, limit) {
  const { rows } = await q(
    `SELECT id, sender, preview, verdict, risk, reason, detail, created_at
       FROM scans WHERE user_id = $1 AND source = $2
      ORDER BY created_at DESC, id DESC LIMIT $3`,
    [String(userId), source, limit]);
  return rows;
}

// Scoped by user_id as well as id: the id comes from the browser, so without it
// anyone could mark anyone else's scans.
export async function markFeedback(userId, id, feedback) {
  const { rows } = await q(
    `UPDATE scans SET feedback = $3, feedback_at = now()
      WHERE id = $1 AND user_id = $2 RETURNING id, source, sender, detail->>'dmarc' AS dmarc`,
    [id, userId, feedback]);
  return rows[0] ?? null;
}

// The one address a From line names, exactly as behaviour.addr_parts reads it
// (the <...> one first), so a vouch and its undo hit the same row the scan uses.
export function addressOf(sender) {
  const s = String(sender ?? "");
  const m = /<([^>]+)>/.exec(s) ?? /([^\s<>,;]+@[^\s<>,;]+)/.exec(s);
  return m ? m[1].toLowerCase().trim() : "";
}

// Undo a "not a threat" vouch (app/api/feedback) once the user calls the same
// sender a scam or spam: same rule as forgetContacts, for one address.
export async function uncontact(userId, sender) {
  const addr = addressOf(sender);
  if (!addr) return;
  try {
    await q(`DELETE FROM sender_profiles WHERE user_id = $1 AND sender = $2 AND contacted AND n = 0`,
            [String(userId), addr]);
    await q(`UPDATE sender_profiles SET contacted = false WHERE user_id = $1 AND sender = $2`,
            [String(userId), addr]);
  } catch (e) {
    if (e.code !== "42P01" && e.code !== "42703") throw e; // no table / column yet
  }
}

// Same scoping as markFeedback, and for the same reason: the id comes from the
// browser, so without user_id in the WHERE anyone could delete anyone's history.
export async function deleteScan(userId, id) {
  const { rows } = await q(
    `DELETE FROM scans WHERE id = $1 AND user_id = $2 RETURNING id`,
    [id, userId]);
  return rows[0] ?? null;
}

// Gmail rows live on their own page (/gmail), so the dashboard's list skips
// them. stats() below still counts them: they are scans all the same.
export async function recentScans(userId, limit = 25) {
  const { rows } = await q(
    // detail->>'label' rather than the whole JSONB: BandBadge needs the one
    // key, and this table renders 25 rows that use nothing else from it.
    `SELECT id, channel, preview, verdict, risk, reason, created_at,
            detail->>'label' AS label
       FROM scans WHERE user_id = $1 AND source <> 'gmail'
      ORDER BY created_at DESC LIMIT $2`,
    [userId, limit]);
  return rows;
}

export async function stats(userId) {
  const { rows } = await q(
    `SELECT COUNT(*)::int AS total,
            COUNT(*) FILTER (WHERE verdict = 'PHISHING')::int   AS phishing,
            COUNT(*) FILTER (WHERE verdict = 'SUSPICIOUS')::int AS suspicious,
            COUNT(*) FILTER (WHERE verdict = 'SAFE')::int       AS safe,
            COUNT(*) FILTER (WHERE channel = 'sms')::int        AS sms,
            COUNT(*) FILTER (WHERE channel = 'email')::int      AS email,
            COUNT(*) FILTER (WHERE channel = 'voice')::int      AS voice
       FROM scans WHERE user_id = $1`, [userId]);
  return rows[0];
}
