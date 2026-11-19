"use client";

import { useRef, useState } from "react";
import { FileAudio, Loader2, RotateCcw, ScanLine } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { VerdictPanel } from "@/components/verdict-panel";
import { RiskTimeline } from "@/components/risk-timeline";

const SAMPLE =
  "Hello sir, I am calling from the SBI credit card department. Your card has been " +
  "used for a transaction of forty-nine thousand rupees in Mumbai. If this was not " +
  "you, we must block it immediately. For verification, please tell me the OTP that " +
  "has just come to your registered mobile number. Sir, do not disconnect the call " +
  "and do not discuss this with anyone, this is a confidential fraud investigation.";

export default function VoicePage() {
  const [text, setText] = useState("");
  const [caller, setCaller] = useState("");
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const ref = useRef(null);

  async function run(fn) {
    setBusy(true);
    setResult(null);
    try {
      const res = await fn();
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

  const scanText = () =>
    text.trim()
      ? run(() => fetch("/api/scan", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ channel: "voice", text, caller }),
        }))
      : toast.error("Paste a transcript first");

  const scanFile = () => {
    if (!file) return toast.error("Choose an audio file first");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("caller", caller);
    return run(() => fetch("/api/voice", { method: "POST", body: fd }));
  };

  return (
    <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">VISHING</p>
          <h1 className="k-title mt-2 text-4xl font-semibold md:text-5xl">Voice Check</h1>
          <p className="text-sm text-muted-foreground">
            Vishing detection: Whisper transcription, scam-act and keyword layers
            over a rolling window, and a wav2vec2 synthetic-voice head.
          </p>
        </div>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Call</CardTitle>
            <CardDescription className="text-xs">
              Upload a recording, or paste a transcript if you already have one.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="file">
              <TabsList className="w-full">
                <TabsTrigger value="file" className="flex-1">Audio file</TabsTrigger>
                <TabsTrigger value="text" className="flex-1">Transcript</TabsTrigger>
              </TabsList>

              <TabsContent value="file" className="space-y-4 pt-4">
                <div
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => { e.preventDefault(); setFile(e.dataTransfer.files?.[0] ?? null); }}
                  className="flex flex-col items-center gap-3 rounded-lg border border-dashed p-8 text-center"
                >
                  <FileAudio className="size-7 text-muted-foreground" />
                  <div className="space-y-1">
                    <p className="text-sm font-medium">
                      {file ? file.name : "Drop a call recording"}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      wav, mp3, m4a, flac · 8 kHz phone audio is fine
                    </p>
                  </div>
                  <input
                    ref={ref} type="file" accept="audio/*" className="hidden"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  />
                  <Button variant="outline" size="sm" onClick={() => ref.current?.click()}>
                    Choose file
                  </Button>
                </div>
                <Button onClick={scanFile} disabled={busy} className="w-full">
                  {busy ? <><Loader2 className="animate-spin" /> Transcribing and scoring…</>
                        : <><ScanLine /> Analyse recording</>}
                </Button>
                {busy && (
                  <p className="text-center text-xs text-muted-foreground">
                    Whisper runs over the whole recording — roughly 20–60s for a few
                    minutes of audio. Keep this tab open.
                  </p>
                )}
              </TabsContent>

              <TabsContent value="text" className="space-y-4 pt-4">
                <Textarea
                  rows={10}
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste the call transcript here…"
                  className="resize-none"
                />
                <div className="flex gap-2">
                  <Button onClick={scanText} disabled={busy} className="flex-1">
                    {busy ? <><Loader2 className="animate-spin" /> Scanning…</>
                          : <><ScanLine /> Scan transcript</>}
                  </Button>
                  <Button variant="outline" onClick={() => setText(SAMPLE)}>Sample</Button>
                  <Button variant="ghost" title="Clear everything"
                          onClick={() => { setText(""); setCaller(""); setFile(null); setResult(null); }}>
                    <RotateCcw />
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Transcript mode skips the synthetic-voice head — with no audio there
                  is no spoof evidence, and that is scored as absent, not as clean.
                </p>
              </TabsContent>
            </Tabs>

            <div className="mt-4 space-y-2">
              <Label htmlFor="caller" className="text-xs">
                Caller&apos;s number <span className="text-muted-foreground">(optional)</span>
              </Label>
              <Input
                id="caller"
                value={caller}
                onChange={(e) => setCaller(e.target.value)}
                placeholder="+91 98765 43210, 1600123456, or withheld"
              />
              <p className="text-xs text-muted-foreground">
                Banks and other financial companies must call from 1600-series
                numbers. Leave blank and the caller checks stay silent.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-4">
        {result ? (
          <>
            <VerdictPanel result={result} />
            <RiskTimeline
              timeline={result.timeline}
              alertAt={result.alert_at_s}
              peak={result.peak_risk}
            />
            {result.transcript && (
              <Card>
                <CardHeader className="pb-2"><CardTitle className="text-sm">Transcript</CardTitle></CardHeader>
                <CardContent>
                  <p className="max-h-64 overflow-y-auto text-sm leading-relaxed text-muted-foreground">
                    {result.transcript}
                  </p>
                </CardContent>
              </Card>
            )}
          </>
        ) : (
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
