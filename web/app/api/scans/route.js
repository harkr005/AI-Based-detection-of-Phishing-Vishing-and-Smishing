import { deleteScan, feedScans, gmailScan, recentScans, stats } from "@/lib/db";
import { userId } from "@/lib/auth";

const GMAIL_SHOWN = 50;

export async function GET(req) {
  const uid = await userId();
  if (!uid) return Response.json({ error: "Unauthorised" }, { status: 401 });

  // ?source=gmail is the inbox poller's page: the newest GMAIL_SHOWN rather
  // than everything ever scanned.
  const params = new URL(req.url).searchParams;
  const source = params.get("source");
  // ?source=gmail&id=N: one row, for the mail's own page. Scoped by user_id.
  if (source === "gmail" && params.has("id")) {
    const id = Number(params.get("id"));
    if (!Number.isInteger(id)) return Response.json({ error: "Bad id" }, { status: 400 });
    try {
      const row = await gmailScan(uid, id);
      return row ? Response.json({ row }) : Response.json({ error: "No such Gmail scan." }, { status: 404 });
    } catch (e) {
      return Response.json({ error: e.message }, { status: 500 });
    }
  }
  if (source === "gmail") {
    try {
      return Response.json({ rows: await feedScans(uid, source, GMAIL_SHOWN), keep: GMAIL_SHOWN });
    } catch (e) {
      return Response.json({ rows: [], keep: GMAIL_SHOWN, error: e.message });
    }
  }

  try {
    const [rows, counts] = await Promise.all([recentScans(uid), stats(uid)]);
    return Response.json({ rows, counts });
  } catch (e) {
    // Home page renders empty-with-a-warning instead of a 500 wall when
    // Postgres is not up yet.
    return Response.json({ rows: [], counts: null, error: e.message });
  }
}

// One row, from the dashboard's delete button. The sender profile built from
// that message is deliberately NOT rolled back: it is a per-sender aggregate,
// not a copy of the message, and unwinding one event from it is not something
// behaviour.py can do.
export async function DELETE(req) {
  const uid = await userId();
  if (!uid) return Response.json({ error: "Unauthorised" }, { status: 401 });

  const id = new URL(req.url).searchParams.get("id");
  if (!id) return Response.json({ error: "No scan to delete" }, { status: 400 });

  try {
    const row = await deleteScan(uid, id);
    if (!row) return Response.json({ error: "Scan not found" }, { status: 404 });
    return Response.json({ ok: true });
  } catch (e) {
    return Response.json({ error: e.message }, { status: 500 });
  }
}
