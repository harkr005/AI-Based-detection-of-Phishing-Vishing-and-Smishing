import {
  createCipheriv, createDecipheriv, createHash, createHmac, randomBytes,
  scryptSync, timingSafeEqual,
} from "node:crypto";

// ponytail: scrypt + an HMAC-signed token, both out of node:crypto. bcrypt and
// next-auth would each be a dependency (and a native build on Windows) for what
// the standard library already does correctly.
//
// Kept free of next/headers so it stays runnable outside a request — see
// lib/hash.test.mjs.

export const SESSION_MAX_AGE = 60 * 60 * 24 * 7; // one week

/* ---- password hashing --------------------------------------------------- */

export function hashPassword(password) {
  const salt = randomBytes(16).toString("hex");
  return `${salt}:${scryptSync(password, salt, 64).toString("hex")}`;
}

export function checkPassword(password, stored) {
  const [salt, hash] = String(stored).split(":");
  if (!salt || !hash) return false;
  const a = Buffer.from(hash, "hex");
  const b = scryptSync(password, salt, a.length);
  // Constant-time: a wrong password must not be distinguishable by timing.
  return a.length === b.length && timingSafeEqual(a, b);
}

/* ---- session token ------------------------------------------------------ */

const sign = (secret, data) =>
  createHmac("sha256", secret).update(data).digest("base64url");

export function makeToken(secret, uid, maxAge = SESSION_MAX_AGE) {
  const data = `${uid}.${Math.floor(Date.now() / 1000) + maxAge}`;
  return `${data}.${sign(secret, data)}`;
}

// Returns the user id, or null if the token was tampered with or has expired.
export function readToken(secret, value) {
  const [uid, exp, mac] = String(value || "").split(".");
  if (!uid || !exp || !mac) return null;
  const expected = sign(secret, `${uid}.${exp}`);
  if (mac.length !== expected.length) return null;
  if (!timingSafeEqual(Buffer.from(mac), Buffer.from(expected))) return null;
  if (Number(exp) < Date.now() / 1000) return null;
  return uid;
}

// Gmail OAuth's `state`: a ten-minute token naming the account, under a key of
// its own. Signed with AUTH_SECRET itself it would BE a session cookie, and
// `state` travels through Google's URLs, server logs and browser history.
const stateKey = (secret) => `oauth-state:${secret}`;
export const makeState = (secret, uid) => makeToken(stateKey(secret), uid, 600);
export const readState = (secret, value) => readToken(stateKey(secret), value);

/* ---- sealed secrets ------------------------------------------------------ */

// A Gmail refresh token is standing read access to someone's mailbox, so it is
// stored encrypted: a dump of the database alone does not hand it over.
// AES-256-GCM, keyed off AUTH_SECRET — rotating that secret already signs
// everyone out, and now it also disconnects every Gmail, which is the same kind
// of event.
const sealKey = (secret) => createHash("sha256").update(`seal:${secret}`).digest();

export function seal(secret, text) {
  const iv = randomBytes(12);
  const c = createCipheriv("aes-256-gcm", sealKey(secret), iv);
  const body = Buffer.concat([c.update(text, "utf8"), c.final()]);
  return [iv, c.getAuthTag(), body].map((b) => b.toString("base64url")).join(".");
}

// Throws on a tampered value or the wrong secret — GCM's tag makes both loud.
export function unseal(secret, sealed) {
  const [iv, tag, body] = String(sealed).split(".").map((s) => Buffer.from(s, "base64url"));
  const d = createDecipheriv("aes-256-gcm", sealKey(secret), iv);
  d.setAuthTag(tag);
  return Buffer.concat([d.update(body), d.final()]).toString("utf8");
}
