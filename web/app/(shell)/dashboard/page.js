"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { MessageSquare, Mail, PhoneCall, ShieldAlert, ShieldQuestion, ShieldCheck, Activity, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { BandBadge } from "@/components/band-badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";

const CHANNELS = [
  { href: "/sms",   label: "SMS Check",   icon: MessageSquare, key: "sms",
    blurb: "MobileBERT ham / spam / smishing, plus link scoring and DLT sender rules." },
  { href: "/email", label: "Email Check", icon: Mail, key: "email",
    blurb: "DistilBERT with a persuasion head, plus §7 sender-behaviour deviation." },
  { href: "/voice", label: "Voice Check", icon: PhoneCall, key: "voice",
    blurb: "Whisper transcription, scam-act detection and the synthetic-voice head." },
];

function Stat({ icon: Icon, label, value, tone }) {
  return (
    <Card>
      <CardContent className="flex items-start justify-between gap-3 pt-6">
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
          <p className={`font-mono text-3xl font-semibold tabular-nums ${tone ?? ""}`}>
            {value}
          </p>
        </div>
        <Icon className={`size-5 shrink-0 ${tone ?? "text-muted-foreground"}`} />
      </CardContent>
    </Card>
  );
}

export default function Home() {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(null);

  const load = () =>
    fetch("/api/scans").then((r) => r.json()).then(setData).catch(() => setData({ rows: [] }));

  useEffect(() => { load(); }, []);

  // Refetch rather than splice the row out: the stat cards above are counted by
  // Postgres, so dropping it locally would leave them one scan ahead.
  async function remove(id) {
    if (!confirm("Delete this scan from your history? This cannot be undone.")) return;
    setBusy(id);
    try {
      const res = await fetch(`/api/scans?id=${id}`, { method: "DELETE" });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error);
      await load();
    } catch (e) {
      toast.error(e.message);
    } finally {
      setBusy(null);
    }
  }

  const c = data?.counts;

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">OVERVIEW</p>
          <h1 className="k-title mt-2 text-4xl font-semibold md:text-5xl">Welcome back</h1>
        <p className="text-sm text-muted-foreground">
          Unified phishing, smishing and vishing detection — content models, link
          analysis and sender behaviour fused into one explainable verdict.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {data ? (
          <>
            <Stat icon={Activity} label="Total scans" value={c?.total ?? 0} />
            <Stat icon={ShieldAlert} label="Malicious" value={c?.phishing ?? 0} tone="band-PHISHING" />
            <Stat icon={ShieldQuestion} label="Suspicious" value={c?.suspicious ?? 0} tone="band-SUSPICIOUS" />
            <Stat icon={ShieldCheck} label="Safe" value={c?.safe ?? 0} tone="band-SAFE" />
          </>
        ) : (
          Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-28" />)
        )}
      </div>

      {data?.error && (
        <Card className="border-amber-500/40 bg-amber-500/5">
          <CardContent className="pt-6 text-sm">
            <span className="band-SUSPICIOUS font-medium">Scan history unavailable.</span>{" "}
            <span className="text-muted-foreground">{data.error}</span>
            <p className="mt-1 text-xs text-muted-foreground">
              Detection still works — only the history table needs Postgres.
            </p>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        {CHANNELS.map(({ href, label, icon: Icon, blurb, key }) => (
          <Link key={href} href={href} className="group">
            <Card className="h-full transition-colors group-hover:border-primary/60">
              <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                  <div className="flex size-8 items-center justify-center rounded-md bg-primary/10">
                    <Icon className="size-4 text-primary" />
                  </div>
                  <CardTitle className="text-sm">{label}</CardTitle>
                  <Badge variant="secondary" className="ml-auto font-mono text-xs">
                    {c?.[key] ?? 0}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-xs leading-relaxed text-muted-foreground">{blurb}</p>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      <Card>
        <CardHeader className="pb-2"><CardTitle className="text-sm">Recent activity</CardTitle></CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-24">Channel</TableHead>
                <TableHead>Preview</TableHead>
                <TableHead className="w-28">Verdict</TableHead>
                <TableHead className="w-20 text-right">Risk</TableHead>
                <TableHead className="w-40">When</TableHead>
                <TableHead className="w-12 sr-only">Delete</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {!data ? (
                <TableRow><TableCell colSpan={6}><Skeleton className="h-5" /></TableCell></TableRow>
              ) : data.rows.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="py-8 text-center text-sm text-muted-foreground">
                    No scans yet — start with an SMS, an email or a call recording.
                  </TableCell>
                </TableRow>
              ) : data.rows.map((r) => (
                <TableRow key={r.id}>
                  <TableCell className="uppercase text-xs text-muted-foreground">{r.channel}</TableCell>
                  <TableCell className="max-w-0 truncate text-sm">{r.preview}</TableCell>
                  <TableCell>
                    <BandBadge verdict={r.verdict} label={r.label} />
                  </TableCell>
                  <TableCell className="text-right font-mono text-sm tabular-nums">
                    {r.risk.toFixed(3)}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {new Date(r.created_at).toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-8 text-muted-foreground hover:text-destructive"
                      disabled={busy === r.id}
                      onClick={() => remove(r.id)}
                      aria-label={`Delete this ${r.channel} scan`}
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
