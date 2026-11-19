"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Inbox } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import GmailCard from "@/components/gmail-card";
import { ScanRow, togglePick } from "@/components/scan-row";

// The connected Gmail's feed — same shape as the Phone SMS page. The server
// polls Gmail every minute (lib/gmail.js); this page re-reads the table every
// fifteen seconds, so a new mail shows up here at most ~75s after it lands.
const POLL_MS = 15_000;

// Mail scanned before subjects were stored has none in detail; its preview
// starts with the subject line, so the first line stands in.
const subjectOf = (r) => r.detail?.subject || r.preview?.split("\n")[0] || "(no subject)";

// Gmail's own spam call, shown beside ours rather than folded into it.
const gmailSpam = (r) => r.detail?.gmail_labels?.includes("SPAM");

export default function GmailPage() {
  const [data, setData] = useState(null);
  const router = useRouter();
  const [picked, setPicked] = useState(() => new Set()); // checked row ids

  useEffect(() => {
    let alive = true;
    const load = () =>
      fetch("/api/scans?source=gmail")
        .then((r) => r.json())
        .then((out) => { if (alive) setData(out); })
        .catch(() => { if (alive) setData({ rows: [] }); });

    load();
    const id = setInterval(load, POLL_MS);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const rows = data?.rows ?? [];

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
          FROM YOUR INBOX
        </p>
        <h1 className="k-title mt-2 text-4xl font-semibold md:text-5xl">Gmail</h1>
        <p className="text-sm text-muted-foreground">
          Every new mail in the connected inbox, scanned as it arrives, newest
          first. The last {data?.keep ?? 50} are shown.
        </p>
      </div>

      <GmailCard onDisconnected={() => setData((d) => ({ ...d, rows: [] }))} />

      {data?.error && (
        <Card className="border-amber-500/40 bg-amber-500/5">
          <CardContent className="pt-6 text-sm">
            <span className="band-SUSPICIOUS font-medium">Feed unavailable.</span>{" "}
            <span className="text-muted-foreground">{data.error}</span>
          </CardContent>
        </Card>
      )}

      {!data ? (
        <div className="space-y-3">
          {Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-16" />)}
        </div>
      ) : rows.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-16 text-center">
            <Inbox className="size-8 text-muted-foreground" />
            <p className="text-sm font-medium">No mail scanned yet.</p>
            <p className="max-w-md text-xs leading-relaxed text-muted-foreground">
              The last ten mails in your inbox are scanned right after you
              connect, and every new one after that. Give it a minute — the
              first scan waits on the model server to warm up.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-2">
          {rows.map((r) => (
            <ScanRow
              key={r.id}
              r={r}
              text={subjectOf(r)}
              badges={gmailSpam(r) && (
                <Badge variant="outline" className="text-xs">Gmail: spam</Badge>
              )}
              picked={picked.has(r.id)}
              onPick={() => setPicked((p) => togglePick(p, r.id))}
              onView={() => router.push(`/gmail/${r.id}`)}
            />
          ))}
        </div>
      )}

    </div>
  );
}
