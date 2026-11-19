import { cookies } from "next/headers";

import { findUserById } from "./db";
import { SESSION_MAX_AGE, makeToken, readToken } from "./hash";

export { checkPassword, hashPassword } from "./hash";

const COOKIE = "sentinel_session";

function secret() {
  const s = process.env.AUTH_SECRET;
  if (!s) throw new Error("AUTH_SECRET is missing from .env.local");
  return s;
}

export async function startSession(uid) {
  (await cookies()).set(COOKIE, makeToken(secret(), uid), {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: SESSION_MAX_AGE,
  });
}

export async function endSession() {
  (await cookies()).delete(COOKIE);
}

// Every API route gates on this. Returns null when signed out.
export async function userId() {
  return readToken(secret(), (await cookies()).get(COOKIE)?.value);
}

export async function currentUser() {
  const uid = await userId();
  if (!uid) return null;
  try {
    return await findUserById(uid);
  } catch {
    // Postgres down: the session is still valid, we just cannot show the name.
    return { id: uid, username: null };
  }
}
