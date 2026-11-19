// node --env-file=.env.local lib/gmail.test.mjs
import assert from "node:assert/strict";

import { deleteGmailScans, forgetContacts, gmailLink, linkGmail, gmailLinks, saveScan } from "./db.js";
import { fullText, pollOne, scanRecent, sealToken, seedProfiles } from "./gmail.js";
import { hashPassword } from "./hash.js";

// The poll loop against the real database, with Google and the sidecar faked.
// pollOne on the test user only, NEVER pollAll: pollAll walks every real
// linked mailbox, and the revoked-grant step below would expire them (it did,
// once, to a real account).
// label filter, pagination, dedup on a retried tick, a vanished message, a
// poison message, and Google revoking the grant.
if (!process.env.DATABASE_URL) {
  console.log("skipped: DATABASE_URL is not set");
  process.exit(0);
}

const pg = new (await import("pg")).default.Pool({ connectionString: process.env.DATABASE_URL });
const cleanup = async () => {
  const { rows } = await pg.query(`DELETE FROM users WHERE username = 'gmail-test' RETURNING id`);
  for (const r of rows) {
    await pg.query(`DELETE FROM scans WHERE user_id = $1`, [String(r.id)]);
    await pg.query(`DELETE FROM sender_profiles WHERE user_id = $1`, [String(r.id)]);
  }
};
await cleanup();
const { rows: [{ id }] } = await pg.query(
  `INSERT INTO users (username, password_hash) VALUES ('gmail-test', $1) RETURNING id`,
  [hashPassword("x")]);
const uid = String(id);
await linkGmail(uid, "test@gmail.com", sealToken("refresh"), "100");

const poll = async () => pollOne((await gmailLinks()).find((u) => u.id === uid));
const added = (mid, labels) => ({ messagesAdded: [{ message: { id: mid, labelIds: labels } }] });
let grant = true, scored = [], seeds = [];
globalThis.fetch = async (url, init) => {
  const u = new URL(url);
  const json = (body, status = 200) => new Response(JSON.stringify(body), { status });
  if (u.pathname === "/token") {
    return grant ? json({ access_token: "a" }) : json({ error: "invalid_grant" }, 400);
  }
  if (u.pathname.endsWith("/history")) {
    assert.equal(u.searchParams.get("startHistoryId"), "100");
    return u.searchParams.get("pageToken")
      ? json({ historyId: "130", history: [added("gone", ["INBOX"]), added("poison", ["INBOX"])] })
      : json({ historyId: "130", nextPageToken: "p2",
               history: [added("m1", ["INBOX", "UNREAD"]), added("sent", ["SENT"]),
                         added("m2", ["SPAM"])] });
  }
  if (u.pathname.endsWith("/messages")) {
    return u.searchParams.get("labelIds") === "SENT"
      ? json({ messages: [{ id: "s1" }] })
      : json({ messages: [{ id: "m4" }, { id: "m3" }, { id: "m1" }] }); // newest first
  }
  if (u.searchParams.get("format") === "metadata") {
    // Headers only, and only the ones asked for: no raw body on this path.
    const mid = u.pathname.split("/").pop();
    const names = u.searchParams.getAll("metadataHeaders");
    return json({ internalDate: "1760000000000",
                  payload: { headers: names.map((n) => ({ name: n, value: `${n}-${mid}` })) } });
  }
  if (u.pathname === "/email/seed") {
    seeds.push(JSON.parse(init.body));
    return json({ senders: 0, contacts: 0 });
  }
  if (u.pathname === "/email/parse") return json({ text: `full-${JSON.parse(init.body).raw}` });
  if (u.pathname.endsWith("/messages/gone")) return json({ error: "not found" }, 404);
  if (u.pathname.includes("/messages/")) {
    const mid = u.pathname.split("/").pop();
    return json({ raw: mid, labelIds: mid === "m2" ? ["SPAM"] : ["INBOX"] });
  }
  if (u.pathname === "/email/raw") {
    const { raw } = JSON.parse(init.body);
    if (raw === "poison") return new Response("no text part", { status: 422 });
    scored.push(raw);
    return json({ verdict: "SAFE", risk: 0.1, reason: "r", sender: `s-${raw}`,
                  preview: `p-${raw}` });
  }
  throw new Error(`unexpected fetch ${url}`);
};

try {
  await poll();
  const { rows } = await pg.query(
    `SELECT message_hash, source, sender, preview, detail->'gmail_labels' AS labels
       FROM scans WHERE user_id = $1 ORDER BY message_hash`, [uid]);
  assert.deepEqual(rows.map((r) => r.message_hash), ["m1", "m2"], "sent mail is not scanned");
  assert.equal(rows[0].source, "gmail");
  assert.equal(rows[0].sender, "s-m1");
  assert.deepEqual(rows[1].labels, ["SPAM"]);
  assert.deepEqual(scored, ["m1", "m2"], "a vanished or poison message is skipped, not fatal");
  assert.equal((await gmailLinks()).find((u) => u.id === uid).gmail_history_id, "130");
  // ...but who the Sent mail went to is folded into the profiles as a contact.
  assert.deepEqual(seeds, [{ user_id: uid, contacts: ["To-sent", "Cc-sent"] }]);

  // A retried tick from the same point must not score (and profile) twice.
  await pg.query(`UPDATE users SET gmail_history_id = '100' WHERE id = $1`, [id]);
  scored = [];
  await poll();
  assert.deepEqual(scored, [], "already-scanned mail is not re-scored");

  // First scan on connect: oldest first, and m1 (already scanned) is not redone.
  await scanRecent(uid, "a", 3);
  assert.deepEqual(scored, ["m3", "m4"]);

  // Seeding on connect: headers of older inbox mail (skipping the ones
  // scanRecent scores) and Sent recipients. Nothing scored, and only once.
  seeds = []; scored = [];
  await seedProfiles(uid, "a", 1);
  assert.deepEqual(seeds, [{
    user_id: uid,
    events: [{ sender: "From-m3", sent_at: "2025-10-09T08:53:20.000Z" },
             { sender: "From-m1", sent_at: "2025-10-09T08:53:20.000Z" }],
    contacts: ["To-s1", "Cc-s1"] }]);
  assert.deepEqual(scored, [], "seeding never scores");
  await seedProfiles(uid, "a", 1);
  assert.equal(seeds.length, 1, "a reconnect does not seed twice");

  // View reads the full text live, and parse-only: nothing is re-scored.
  scored = [];
  assert.equal((await fullText(sealToken("refresh"), "m1")).text, "full-m1");
  assert.deepEqual(scored, []);

  // Google stops honouring the grant: token gone, address kept for "reconnect".
  grant = false;
  await poll();
  const link = await gmailLink(uid);
  assert.equal(link.gmail_token, null);
  assert.equal(link.gmail_address, "test@gmail.com");
  // A token sealed under an old AUTH_SECRET expires the same way.
  grant = true;
  await linkGmail(uid, "test@gmail.com", "not.a.sealed-token", "100");
  await poll();
  assert.equal((await gmailLink(uid)).gmail_token, null, "an unopenable token expires the link");
  // Disconnect forgets who they wrote to: contact-only rows go, real senders
  // stay but lose the flag.
  await pg.query(`INSERT INTO sender_profiles (user_id, sender, n, contacted)
                  VALUES ($1, 'pal@x.com', 0, true), ($1, 'news@x.com', 4, true)`, [uid]);
  assert.equal(await forgetContacts(uid), 1);
  const { rows: prof } = await pg.query(
    `SELECT sender, contacted FROM sender_profiles WHERE user_id = $1`, [uid]);
  assert.deepEqual(prof, [{ sender: "news@x.com", contacted: false }]);
  // Disconnect deletes this user's Gmail rows and nothing else of theirs.
  await saveScan(uid, "email", "pasted", { verdict: "SAFE", risk: 0.1 });
  assert.equal(await deleteGmailScans(uid), 4); // m1..m4
  const { rows: left } = await pg.query(`SELECT source FROM scans WHERE user_id = $1`, [uid]);
  assert.deepEqual(left.map((r) => r.source), ["web"], "pasted scans survive a disconnect");
  console.log("gmail poll self-check passed");
} finally {
  await cleanup();
  await pg.end();
  await globalThis._pgPool.end(); // db.js's own pool, or node never exits
}
