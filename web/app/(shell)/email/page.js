"use client";

import { useState } from "react";
import { Loader2, RotateCcw, ScanLine } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { OcrInput } from "@/components/ocr-input";
import { VerdictPanel } from "@/components/verdict-panel";

const SAMPLE = {
  sender: "HDFC Bank Security <alerts@hdfc-secure-in.tk>",
  subject: "Immediate action required: your account will be suspended",
  body:
    "Dear valued customer,\n\nWe have detected unusual sign-in activity on your " +
    "account. For your protection your net banking access will be suspended " +
    "within 24 hours unless you re-verify your identity.\n\n" +
    "Verify now: http://hdfc-secure-in.tk/verify?id=8842\n\n" +
    "Failure to act will result in permanent closure.\n\nHDFC Bank Security Team",
};

export default function EmailPage() {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [sender, setSender] = useState("");
  // See the SMS page: only OCR (or, later, the Gmail push) carries a real send
  // time, and hour_dev / burst_1h abstain without one.
  const [sentAt, setSentAt] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  async function scan() {
    if (!body.trim()) return toast.error("Paste the email body first");
    setBusy(true);
    setResult(null);
    try {
      const res = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ channel: "email", subject, body, sender,
                               sent_at: sentAt || null }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Scan failed");
      setResult(data);
      if (data.saveError) toast.warning(`Not saved to history: ${data.saveError}`);
    } catch (e) {
      toast.error(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">PHISHING</p>
          <h1 className="k-title mt-2 text-4xl font-semibold md:text-5xl">Email Check</h1>
          <p className="text-sm text-muted-foreground">
            Phishing detection: DistilBERT with a persuasion-principle head, link
            scoring, and sender-behaviour deviation.
          </p>
        </div>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Message</CardTitle>
            <CardDescription className="text-xs">
              Paste the email, or drop a screenshot of it.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="from" className="text-xs">
                From <span className="text-muted-foreground">(optional)</span>
              </Label>
              <Input
                id="from"
                value={sender}
                onChange={(e) => setSender(e.target.value)}
                placeholder="Display Name &lt;user@domain.com&gt;"
              />
              <p className="text-xs text-muted-foreground">
                Enables display-name / domain mismatch and first-contact checks.
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="subject" className="text-xs">Subject</Label>
              <Input
                id="subject"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="Subject line"
              />
            </div>

            <Tabs defaultValue="text">
              <TabsList className="w-full">
                <TabsTrigger value="text" className="flex-1">Body</TabsTrigger>
                <TabsTrigger value="ocr" className="flex-1">Screenshot (OCR)</TabsTrigger>
              </TabsList>
              <TabsContent value="text" className="pt-4">
                <Textarea
                  rows={10}
                  value={body}
                  onChange={(e) => setBody(e.target.value)}
                  placeholder="Paste the email body here…"
                  className="resize-none"
                />
              </TabsContent>
              <TabsContent value="ocr" className="pt-4">
                <OcrInput
                  channel="email"
                  onExtract={({ sender, subject, body, sent_at }) => {
                    setBody(body);
                    if (sender) setSender(sender);
                    if (subject) setSubject(subject);
                    setSentAt(sent_at || "");
                  }}
                  label="Screenshot of the email"
                />
                {body && (
                  <Textarea
                    rows={8}
                    value={body}
                    onChange={(e) => setBody(e.target.value)}
                    className="mt-4 resize-none"
                  />
                )}
              </TabsContent>
            </Tabs>

            <div className="flex gap-2">
              <Button onClick={scan} disabled={busy} className="flex-1">
                {busy ? <><Loader2 className="animate-spin" /> Scanning…</>
                      : <><ScanLine /> Scan email</>}
              </Button>
              <Button
                variant="outline"
                onClick={() => { setSender(SAMPLE.sender); setSubject(SAMPLE.subject);
                                 setBody(SAMPLE.body); setSentAt(""); }}
              >
                Sample
              </Button>
              <Button variant="ghost" title="Clear everything"
                      onClick={() => { setSender(""); setSubject(""); setBody("");
                                       setSentAt(""); setResult(null); }}>
                <RotateCcw />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-4">
        {result ? <VerdictPanel result={result} /> : (
          <Card className="border-dashed">
            <CardContent className="flex h-40 items-center justify-center text-sm text-muted-foreground">
              Results appear here.
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
