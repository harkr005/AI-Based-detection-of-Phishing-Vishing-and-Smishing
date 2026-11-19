"use client";

import { useEffect, useState } from "react";
import { Check, Mail } from "lucide-react";
import { toast } from "sonner";

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

// Connects a Gmail inbox. The server polls it every minute (lib/gmail.js) and
// new mail lands in the feed on the /gmail page.
// onDisconnected: the page's feed empties at once instead of on its next poll.
export default function GmailCard({ onDisconnected }) {
  const [status, setStatus] = useState(null); // { address, connected }
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const load = () =>
    fetch("/api/gmail")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus({}));

  useEffect(() => {
    load();
    // The OAuth callback lands back on this page with ?gmail=<outcome>.
    const msg = new URLSearchParams(window.location.search).get("gmail");
    if (msg) {
      if (msg === "connected") toast.success("Gmail connected.");
      else toast.error(msg, { duration: 10_000 });
      window.history.replaceState(null, "", window.location.pathname);
    }
  }, []);

  async function disconnect() {
    if (!confirm(
      "Disconnect Gmail?\n\n" +
      "Every scanned Gmail message will be DELETED from Sentinel, with its " +
      "verdict, and so will the list of people you have written to. This " +
      "cannot be undone.\n\n" +
      "The senders Sentinel has learned stay, so pasted mail is still judged " +
      "against them. Your mail in Gmail itself is not touched.",
    )) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/gmail", { method: "DELETE" });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error);
      toast.success(`Gmail disconnected. ${out.deleted} scanned message${out.deleted === 1 ? "" : "s"} deleted.`);
      onDisconnected?.();
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const connect = (label) => (
    <Button size="lg" className="px-5" asChild>
      <a href="/api/gmail/connect">{label}</a>
    </Button>
  );

  // One row in every state: the icon, then whatever the state needs beside it.
  return (
    <Card size="sm">
      <CardContent className="space-y-2">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-md bg-primary/10">
            <Mail className="size-4 text-primary" />
          </div>
          {!status ? (
            <p className="text-sm text-muted-foreground">Checking…</p>
          ) : status.connected ? (
            <>
              <p className="flex min-w-0 items-center gap-1.5 text-sm">
                <Check className="size-4 shrink-0 band-SAFE" />
                <span className="band-SAFE font-medium">Connected</span>
                <span className="truncate text-muted-foreground">· {status.address}</span>
              </p>
              <Button size="lg" variant="ghost" disabled={busy}
                      className="ml-auto text-muted-foreground hover:text-destructive"
                      onClick={disconnect}>
                Disconnect
              </Button>
            </>
          ) : status.address ? (
            <>
              {connect("Reconnect")}
              <p className="text-sm">
                <span className="band-SUSPICIOUS font-medium">Access expired</span>{" "}
                <span className="text-muted-foreground">for {status.address}</span>
              </p>
              <Button size="lg" variant="ghost" disabled={busy}
                      className="ml-auto text-muted-foreground" onClick={disconnect}>
                Forget
              </Button>
            </>
          ) : (
            connect("Connect Gmail")
          )}
        </div>

        {/* Disclosure for lib/gmail.js seedProfiles and the Sent poll. */}
        <p className="text-xs text-muted-foreground">
          To learn who you know, Sentinel reads the sender of about 200 older inbox
          mails and who your sent mail goes to — headers only, never the text.
          Who you write to is forgotten when you disconnect.
        </p>
        {error && <p className="text-xs band-PHISHING">{error}</p>}
      </CardContent>
    </Card>
  );
}
