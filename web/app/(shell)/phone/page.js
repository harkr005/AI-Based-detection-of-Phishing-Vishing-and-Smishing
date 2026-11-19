import { Download, ShieldCheck, Smartphone } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// The Android app's download page. The app scores SMS entirely on the phone and
// has no internet permission, so there is no pairing and nothing from it ever
// appears on this site. The APK is copied into public/ by hand after a build
// (see README) and is gitignored: it carries the SMS model's weights.
const APK = "/sentinel.apk";

const STEPS = [
  {
    title: "Download the app",
    body: "Click the button above. The file is saved on this computer.",
  },
  {
    title: "Send it to your phone",
    body: "Copy sentinel.apk over a USB cable, Google Drive or email to yourself. With USB debugging on you can also run: adb install sentinel.apk",
  },
  {
    title: "Install it",
    body: "Open the file on the phone. If Android blocks it, allow \"Install unknown apps\" for the app you opened it from (Files, Chrome or Drive), then tap Install.",
  },
  {
    title: "Allow SMS access",
    body: "Open Sentinel and tap Allow when it asks to read SMS. On Android 13 and newer the request can be blocked silently: go to Settings → Apps → Sentinel, tap ⋮ → Allow restricted settings, then Permissions → SMS → Allow.",
  },
  {
    title: "That's it",
    body: "Your recent messages are scanned straight away, and every new SMS is checked as it arrives. Tap any message to see why it was flagged.",
  },
];

export default function PhonePage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
          ANDROID APP
        </p>
        <h1 className="k-title mt-2 text-4xl font-semibold md:text-5xl">Sentinel for your phone</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Checks every SMS for scams the moment it arrives, using the same SMS and
          link models as this website, running on the phone itself.
        </p>
      </div>

      <Card>
        <CardContent className="flex flex-col gap-4 pt-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <Smartphone className="size-8 text-primary" />
            <div>
              <p className="font-medium">sentinel.apk</p>
              <p className="text-xs text-muted-foreground">Android 7.0 or newer</p>
            </div>
          </div>
          <Button asChild size="lg">
            <a href={APK} download>
              <Download className="size-4" /> Download APK
            </a>
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>How to set it up</CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="space-y-4">
            {STEPS.map((s, i) => (
              <li key={s.title} className="flex gap-3">
                <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-primary/15 text-xs font-semibold text-primary">
                  {i + 1}
                </span>
                <div>
                  <p className="text-sm font-medium">{s.title}</p>
                  <p className="text-sm leading-relaxed text-muted-foreground">{s.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </CardContent>
      </Card>

      <Card className="border-emerald-500/30 bg-emerald-500/5">
        <CardContent className="flex gap-3 pt-6 text-sm">
          <ShieldCheck className="size-5 shrink-0 text-emerald-500" />
          <p className="leading-relaxed text-muted-foreground">
            <span className="font-medium text-foreground">Your messages stay on your phone.</span>{" "}
            The app has no internet access: messages are scanned on the device and
            nothing, not even the verdict, is sent anywhere. That is why phone
            scans do not appear on this website.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
