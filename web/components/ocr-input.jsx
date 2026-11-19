"use client";

import { useRef, useState } from "react";
import { ImageUp, Loader2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";

/** Screenshot -> fields via Gemini (§5's "forum mining: screenshots -> OCR").
 *  Hands the parsed parts (sms: sender/text, email: sender/subject/body) back to
 *  the parent, which drops them in the form so the user can correct OCR slips
 *  before scanning. */
export function OcrInput({ channel, onExtract, label = "Scan a screenshot" }) {
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState(null);
  const ref = useRef(null);

  async function handle(file) {
    if (!file) return;
    setBusy(true);
    setPreview(URL.createObjectURL(file));
    try {
      const fd = new FormData();
      fd.append("image", file);
      fd.append("channel", channel);
      const res = await fetch("/api/ocr", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "OCR failed");
      if (!Object.values(data).some((v) => v?.trim())) {
        toast.warning("No text found in that image");
        return;
      }
      onExtract(data);
      toast.success("Text extracted — check it before scanning");
    } catch (e) {
      toast.error(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); handle(e.dataTransfer.files?.[0]); }}
      className="flex flex-col items-center gap-3 rounded-lg border border-dashed p-6 text-center"
    >
      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={preview} alt="" className="max-h-40 rounded-md border object-contain" />
      ) : (
        <ImageUp className="size-7 text-muted-foreground" />
      )}
      <div className="space-y-1">
        <p className="text-sm font-medium">{label}</p>
        <p className="text-xs text-muted-foreground">
          Drop a screenshot, or pick a file. Gemini extracts the text.
        </p>
      </div>
      <input
        ref={ref}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handle(e.target.files?.[0])}
      />
      <Button variant="outline" size="sm" disabled={busy} onClick={() => ref.current?.click()}>
        {busy ? <><Loader2 className="animate-spin" /> Reading…</> : "Choose image"}
      </Button>
    </div>
  );
}
