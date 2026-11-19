// Runs once when the Next server starts. Its only job is the Gmail poller —
// a Next app has no worker process, and this is the one hook that runs outside
// any request.
export async function register() {
  if (process.env.NEXT_RUNTIME !== "nodejs") return;
  if (process.env.NEXT_PHASE === "phase-production-build") return;
  if (!process.env.GOOGLE_CLIENT_ID) return; // Gmail not configured: nothing to poll
  if (globalThis._gmailTimer) return;

  const { POLL_MS, pollAll } = await import("./lib/gmail");
  // unref: the timer must never be the thing keeping a process alive.
  globalThis._gmailTimer = setInterval(pollAll, POLL_MS).unref();
}
