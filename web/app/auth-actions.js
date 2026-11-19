"use server";

import { redirect } from "next/navigation";

import { createUser, findUser } from "@/lib/db";
import { checkPassword, endSession, hashPassword, startSession } from "@/lib/auth";
import { passwordProblem, usernameProblem } from "@/lib/password";

// The client form checks the same rules for instant feedback; these are the
// ones that actually decide. redirect() throws a control-flow signal, so it
// always sits outside the try.

export async function signUp(_prev, formData) {
  const username = String(formData.get("username") || "").trim();
  const password = String(formData.get("password") || "");

  const bad = usernameProblem(username) || passwordProblem(password);
  if (bad) return { error: bad };

  let user;
  try {
    user = await createUser(username, hashPassword(password));
  } catch (e) {
    return { error: `Could not reach the database: ${e.message}` };
  }
  if (!user) return { error: "That username is already taken." };

  await startSession(user.id);
  redirect("/dashboard");
}

export async function signIn(_prev, formData) {
  const username = String(formData.get("username") || "").trim();
  const password = String(formData.get("password") || "");
  if (!username || !password) return { error: "Enter your username and password." };

  let user;
  try {
    user = await findUser(username);
  } catch (e) {
    return { error: `Could not reach the database: ${e.message}` };
  }

  // Same message either way — telling an attacker which half was wrong tells
  // them which usernames exist.
  if (!user || !checkPassword(password, user.password_hash))
    return { error: "Wrong username or password." };

  await startSession(user.id);
  redirect("/dashboard");
}

export async function signOut() {
  await endSession();
  redirect("/");
}
