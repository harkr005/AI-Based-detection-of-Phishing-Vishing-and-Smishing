// The one password rule, imported by both the form (instant feedback) and the
// server action (the actual gate). Client-side checks are a courtesy; the
// server never trusts them.
export const RULES = [
  { label: "at least 8 characters", ok: (p) => p.length >= 8 },
  { label: "one uppercase letter", ok: (p) => /[A-Z]/.test(p) },
  { label: "one number", ok: (p) => /[0-9]/.test(p) },
];

export function passwordProblem(password) {
  const failed = RULES.filter((r) => !r.ok(password || ""));
  if (!failed.length) return null;
  return `Password needs ${failed.map((r) => r.label).join(", ")}.`;
}

export function usernameProblem(username) {
  const u = (username || "").trim();
  if (u.length < 3) return "Username needs at least 3 characters.";
  if (u.length > 32) return "Username is capped at 32 characters.";
  if (!/^[a-zA-Z0-9._-]+$/.test(u))
    return "Username can use letters, numbers, dot, dash and underscore only.";
  return null;
}
