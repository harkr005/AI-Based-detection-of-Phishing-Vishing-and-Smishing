"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Inbox } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { BandBadge } from "@/components/band-badge";
import { VerdictPanel } from "@/components/verdict-panel";

// One Gmail scan on a page of its own (the side sheet was too narrow for a
// whole mail plus its verdict). The verdict is the stored scan row; the mail
// is fetched live from Gmail each visit and never stored (lib/gmail.js
// fullText), so the two load separately and the verdict survives a Gmail error.

const SHORT = 800; // characters of text shown before "View full message"

// The mail as sent, for looking at, never for interacting with. sandbox=""
// blocks scripts, forms, popups and same-origin access; links arrive with no
// href (data.safe_html) and <base target=_blank> is the backstop, so a click
// goes nowhere (the verdict panel lists where links really lead). The CSP
// blocks fonts and every remote load except images. Images load by default
// (Mohak's call, 2026-10-11) - a remote image is a read receipt for whoever
// sent the mail, the same trade Gmail makes.
// ponytail: fixed-height frame with its own scroll; sizing it to the content
// would need allow-same-origin, which is the one thing this must never get.
function AsSent({ html }) {
  const csp = "default-src 'none'; style-src 'unsafe-inline'; img-src data: https: http:";
  // The server (data.safe_html) already removed links, metas and <link>s; the
  // referrer meta keeps image requests from naming this dashboard.
  const doc = `<meta http-equiv="Content-Security-Policy" content="${csp}">`
    + `<meta name="referrer" content="no-referrer"><base target="_blank">${html}`;
  return (
    <iframe title="The mail as sent" sandbox="" srcDoc={doc} referrerPolicy="no-referrer"
            className="h-[75vh] w-full rounded-md border bg-white" />
  );
}

// HTML mail is shown as sent, only; the text view remains for mail that has
// no HTML part (plain-text mail), collapsed past SHORT characters.
function MessageBody({ full }) {
  const [long, setLong] = useState(false);  // whole text shown
  const cut = full.text.length > SHORT && !long;
  return (
    <>
      {full.subject && <p className="text-base font-semibold">{full.subject}</p>}
      {full.html ? (
        <>
          <AsSent html={full.html} />
          <p className="text-xs text-muted-foreground">
            Scripts, forms and links are switched off here.
          </p>
        </>
      ) : (
        <>
          <p className="whitespace-pre-wrap text-sm leading-relaxed [overflow-wrap:anywhere]">
            {cut ? `${full.text.slice(0, SHORT).trimEnd()}…` : full.text}
          </p>
          {full.text.length > SHORT && (
            <Button size="sm" variant="link" className="px-0" onClick={() => setLong(!long)}>
              {long ? "Show less" : "View full message"}
            </Button>
          )}
        </>
      )}
    </>
  );
}

// { row } | { error } from /api/scans; null while loading. Same shape for full.
function useJson(url) {
  const [out, setOut] = useState(null);
  useEffect(() => {
    let alive = true;
    fetch(url)
      .then(async (res) => {
        const body = await res.json();
        if (alive) setOut(res.ok ? body : { error: body.error || "Could not load this." });
      })
      .catch((e) => { if (alive) setOut({ error: e.message }); });
    return () => { alive = false; };
  }, [url]);
  return out;
}

export default function GmailMessagePage({ params }) {
  const { id } = use(params);
  const scan = useJson(`/api/scans?source=gmail&id=${encodeURIComponent(id)}`);
  const full = useJson(`/api/gmail/message?id=${encodeURIComponent(id)}`);
  const r = scan?.row;

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <Button asChild variant="ghost" size="sm" className="-ml-2">
        <Link href="/gmail"><ArrowLeft /> Back to Gmail</Link>
      </Button>

      {!scan ? (
        <Skeleton className="h-24" />
      ) : scan.error ? (
        <Card className="border-amber-500/40 bg-amber-500/5">
          <CardContent className="pt-6 text-sm">
            <span className="band-SUSPICIOUS font-medium">Can&apos;t open this mail.</span>{" "}
            <span className="text-muted-foreground">{scan.error}</span>
          </CardContent>
        </Card>
      ) : (
        <>
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <Inbox className="size-4 text-primary" />
              <h1 className="text-xl font-semibold [overflow-wrap:anywhere]">{r.sender || "Unknown sender"}</h1>
              <BandBadge verdict={r.verdict} label={r.detail?.label} gmailLabels={r.detail?.gmail_labels} />
            </div>
            <p className="text-xs text-muted-foreground">{new Date(r.created_at).toLocaleString()}</p>
          </div>

          <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">The message</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {full?.text ? (
                  <MessageBody full={full} />
                ) : (
                  <p className="whitespace-pre-wrap text-sm leading-relaxed">{r.preview}</p>
                )}
                <p className={`text-xs ${full?.error ? "band-SUSPICIOUS" : "text-muted-foreground"}`}>
                  {!full ? "Loading the full mail from Gmail…"
                    : full.error ? `${full.error} Showing the stored preview.`
                    : "Read live from Gmail and never stored here, apart from the first 200 characters."}
                </p>
              </CardContent>
            </Card>
            {/* id: detail is the sidecar result, saved before the row had one, and
                the report buttons mark the row. */}
            {r.detail && <VerdictPanel result={{ ...r.detail, id: r.id }} />}
          </div>
        </>
      )}
    </div>
  );
}
