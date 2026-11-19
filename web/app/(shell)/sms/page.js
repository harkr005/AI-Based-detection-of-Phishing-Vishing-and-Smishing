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

const SAMPLE =
  "Dear customer, your SBI KYC is incomplete. Your account will be blocked today. " +
  "Update now at sbi-kyc-verify.tk/update";

export default function SmsPage() {
  const [text, setText] = useState("");
  const [sender, setSender] = useState("");
  // Only OCR sets this. Typed messages carry no send time, and the behaviour
  // layer's odd-hour rule abstains rather than scoring the hour of the scan.
  const [sentAt, setSentAt] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  async function scan() {
    if (!text.trim()) return toast.error("Paste a message first");
    setBusy(true);
    setResult(null);
    try {
      const res = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ channel: "sms", text, sender, sent_at: sentAt || null }),
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
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">SMISHING</p>
          <h1 className="k-title mt-2 text-4xl font-semibold md:text-5xl">SMS Check</h1>
          <p className="text-sm text-muted-foreground">
            Smishing detection: MobileBERT over the message, XGBoost over any link,
            and Indian DLT sender-ID rules.
          </p>
        </div>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Message</CardTitle>
            <CardDescription className="text-xs">
              Type it, or drop a screenshot and let OCR fill it in.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Tabs defaultValue="text">
              <TabsList className="w-full">
                <TabsTrigger value="text" className="flex-1">Text</TabsTrigger>
                <TabsTrigger value="ocr" className="flex-1">Screenshot (OCR)</TabsTrigger>
              </TabsList>
              <TabsContent value="text" className="pt-4">
                <Textarea
                  rows={6}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste the SMS here…"
                  className="resize-none"
                />
              </TabsContent>
              <TabsContent value="ocr" className="pt-4">
                <OcrInput
                  channel="sms"
                  onExtract={({ sender, text, sent_at }) => {
                    setText(text);
                    if (sender) setSender(sender);
                    setSentAt(sent_at || "");
                  }}
                  label="Screenshot of the SMS"
                />
                {text && (
                  <Textarea
                    rows={5}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    className="mt-4 resize-none"
                  />
                )}
              </TabsContent>
            </Tabs>

            <div className="space-y-2">
              <Label htmlFor="sender" className="text-xs">
                Sender ID <span className="text-muted-foreground">(optional)</span>
              </Label>
              <Input
                id="sender"
                value={sender}
                onChange={(e) => setSender(e.target.value)}
                placeholder="AX-HDFCBK, or a 10-digit number"
              />
              <p className="text-xs text-muted-foreground">
                Leave blank and the behaviour channel stays silent — its weight is
                redistributed rather than counted as zero risk.
              </p>
              <p className="text-xs text-muted-foreground">
                {sentAt
                  ? `Sent ${new Date(sentAt).toLocaleString()} — read off the screenshot, so the odd-hour rule can score it.`
                  : "No send time: the odd-hour rule abstains. It only has one when a screenshot shows the timestamp."}
              </p>
            </div>

            <div className="flex gap-2">
              <Button onClick={scan} disabled={busy} className="flex-1">
                {busy ? <><Loader2 className="animate-spin" /> Scanning…</>
                      : <><ScanLine /> Scan message</>}
              </Button>
              <Button variant="outline"
                      onClick={() => { setText(SAMPLE); setSender("VM-681204"); setSentAt(""); }}>
                Sample
              </Button>
              <Button variant="ghost" title="Clear everything"
                      onClick={() => { setText(""); setSender(""); setSentAt(""); setResult(null); }}>
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
