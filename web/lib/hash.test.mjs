// node lib/hash.test.mjs
import assert from "node:assert/strict";

import { checkPassword, hashPassword, makeState, makeToken, readState, readToken, seal, unseal } from "./hash.js";
import { passwordProblem, usernameProblem } from "./password.js";

const S = "test-secret";

// passwords
const stored = hashPassword("Str0ngPass");
assert.ok(!stored.includes("Str0ngPass"), "plaintext must never appear in the hash");
assert.equal(checkPassword("Str0ngPass", stored), true);
assert.equal(checkPassword("str0ngpass", stored), false);
assert.equal(checkPassword("", stored), false);
assert.equal(checkPassword("Str0ngPass", "garbage"), false);
assert.notEqual(hashPassword("same"), hashPassword("same"), "salt must differ per hash");

// the rule the user asked for: 8+, one uppercase, one number
assert.equal(passwordProblem("Str0ngPass"), null);
assert.match(passwordProblem("Sh0rt"), /8 characters/);
assert.match(passwordProblem("alllowercase1"), /uppercase/);
assert.match(passwordProblem("NoDigitsHere"), /number/);
assert.match(passwordProblem("short"), /8 characters, one uppercase letter, one number/);

// usernames
assert.equal(usernameProblem("mohak_01"), null);
assert.match(usernameProblem("ab"), /3 characters/);
assert.match(usernameProblem("has space"), /letters, numbers/);

// session tokens
const t = makeToken(S, "42");
assert.equal(readToken(S, t), "42");
assert.equal(readToken("other-secret", t), null, "a different key must not verify");
assert.equal(readToken(S, t.replace(/.$/, "x")), null, "tampering must not verify");
assert.equal(readToken(S, "42.9999999999.notamac"), null);
assert.equal(readToken(S, makeToken(S, "42", -10)), null, "expired must not verify");
assert.equal(readToken(S, ""), null);

// OAuth state: round-trips, but is never accepted as a session token
assert.equal(readState(S, makeState(S, "42")), "42");
assert.equal(readToken(S, makeState(S, "42")), null, "an OAuth state must not work as a session");
assert.equal(readState(S, makeToken(S, "42")), null, "a session must not work as an OAuth state");

// sealed secrets (the stored Gmail refresh token)
const sealed = seal(S, "1//refresh-token");
assert.ok(!sealed.includes("refresh"), "plaintext must never appear sealed");
assert.equal(unseal(S, sealed), "1//refresh-token");
assert.notEqual(seal(S, "x"), seal(S, "x"), "iv must differ per seal");
assert.throws(() => unseal("other-secret", sealed), "a different key must not open it");
const [iv, tag, body] = sealed.split(".");
assert.throws(() => unseal(S, [iv, tag, body.replace(/^./, (c) => (c === "A" ? "B" : "A"))].join(".")),
              "tampering must not open it");

console.log("hash/password/session/seal self-check passed");
