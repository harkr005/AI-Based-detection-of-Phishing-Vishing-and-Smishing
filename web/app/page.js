import Link from "next/link";
import {
  ArrowRight, MessageSquare, Mail, PhoneCall, ShieldCheck, Eye, Radar,
  Layers, Waves,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { userId } from "@/lib/auth";

const NAV = [
  { href: "#home", label: "Home" },
  { href: "#why", label: "Why Sentinel" },
  { href: "#channels", label: "Channels" },
  { href: "#how", label: "How It Works" },
];

// Every number below is measured in this project — see PROJECT_SPEC.md.
const PILLARS = [
  { icon: Layers, title: "Three channels, one stack",
    body: "Email, SMS and voice share the same fusion engine, the same risk bands and the same report format." },
  { icon: Radar, title: "Sender behaviour, not just text",
    body: "Content models plateau on consent-based spam. Profiling the sender lifts recall from 0.61 to 0.91." },
  { icon: Eye, title: "Every verdict explains itself",
    body: "TreeSHAP on the link model, exact per-word attribution on the SMS model, and a lead line naming what decided the verdict." },
  { icon: Waves, title: "Hardened against evasion",
    body: "Unicode homoglyphs, leetspeak and zero-width padding are normalised at train and serve time alike." },
];

const CHANNELS = [
  { icon: Mail, tag: "Email", title: "Phishing", metric: "0.986", unit: "phishing recall",
    body: "DistilBERT with a persuasion head, link and attachment checks, then sender-behaviour deviation over the live mail stream. On unseen 2025-26 mail it catches 86.2% of phishing at 1.9% false alarms." },
  { icon: MessageSquare, tag: "SMS", title: "Smishing", metric: "0.870", unit: "cross-source macro-F1",
    body: "A character n-gram model over the message, XGBoost over any link, and Indian DLT sender-ID rules - scored across sources the model never trained on." },
  { icon: PhoneCall, tag: "Voice", title: "Vishing", metric: "96.2%", unit: "unseen scam calls caught",
    body: "Whisper transcription, rolling scam-act detection on a 15-second context, and a wav2vec2 synthetic-voice head that catches 86% of AI voices from tools it never trained on." },
];

const STEPS = [
  { n: "1", title: "Create your account",
    body: "Sign up in seconds. Your scan history is scoped to you and nobody else." },
  { n: "2", title: "Bring the suspicious message",
    body: "Paste an SMS or an email, upload a screenshot for OCR, or drop in a call recording." },
  { n: "3", title: "Read the verdict",
    body: "One risk score, three bands, and the evidence behind it - channel by channel." },
];

const BARS = [
  { label: "Content (MobileBERT)", value: 0.93 },
  { label: "Link (XGBoost)", value: 0.88 },
  { label: "Sender behaviour", value: 0.71 },
];

export default async function Landing() {
  // Reading the session cookie makes this route dynamic, which is the price of
  // the nav knowing whether you are already signed in.
  const signedIn = !!(await userId());
  const cta = signedIn
    ? { href: "/dashboard", label: "Open dashboard" }
    : { href: "/sign-up", label: "Get Started Now" };

  return (
    <div className="k-ambient relative min-h-dvh overflow-x-hidden text-white">
      {/* ---- nav ---- */}
      <nav className="absolute top-6 left-0 z-50 w-full">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between px-6 xl:px-16">
          <Link href="/" className="flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-lg bg-gradient-to-l from-[#ff1e00] to-[#ff3c1c]">
              <ShieldCheck className="size-5" />
            </span>
            <span className="font-heading text-xl font-semibold tracking-tight">Sentinel</span>
          </Link>

          <div className="k-streak hidden rounded-2xl lg:block">
            <div className="flex gap-2 rounded-2xl bg-white/10 p-2">
              {NAV.map(({ href, label }) => (
                <a
                  key={href}
                  href={href}
                  className="rounded-[10px] px-5 py-2.5 font-medium text-[#b2b2b2] transition-all duration-500 hover:bg-[#ff3c1c] hover:text-white"
                >
                  {label}
                </a>
              ))}
            </div>
          </div>

          <div className="k-streak overflow-hidden rounded-2xl">
            <Link
              href={signedIn ? "/dashboard" : "/sign-in"}
              className="flex h-14 items-center rounded-2xl bg-black/40 px-6 font-medium transition-all duration-500 hover:bg-black"
            >
              {signedIn ? "Dashboard" : "Sign in"}
            </Link>
          </div>
        </div>
      </nav>

      {/* ---- hero ---- */}
      <section id="home" className="relative overflow-hidden pt-40 pb-20 lg:pb-40">
        <div className="relative z-10 mx-auto max-w-[1440px] px-6 xl:px-16">
          <div className="grid grid-cols-1 items-center gap-14 lg:grid-cols-2">
            <div>
              <div className="mb-6 w-fit rounded-xl bg-gradient-to-l from-transparent to-white/10 pt-0.5 lg:mb-8">
                <span className="inline-block rounded-xl bg-white/5 px-5 py-2.5 text-sm text-[#f9e7e5] lg:px-6 lg:py-3.5 lg:text-lg">
                  AI-Powered Social-Engineering Defence
                </span>
              </div>

              <h1 className="k-title text-5xl font-semibold lg:text-[80px]">
                Phishing, smishing and vishing, one verdict
              </h1>

              <p className="pt-5 pb-8 text-base text-[#f9e7e5] lg:pt-6 lg:pb-10 lg:text-lg">
                Sentinel reads the message, scores the link, profiles the sender and
                listens for a synthetic voice, then fuses all four into a single
                explainable risk score you can act on.
              </p>

              <div className="flex flex-wrap items-center gap-4">
                <Button asChild size="lg" className="h-14 rounded-xl px-6 text-base">
                  <Link href={cta.href}>
                    {cta.label} <ArrowRight className="ml-1 size-4" />
                  </Link>
                </Button>
                <a href="#how" className="text-[#b3a19f] transition-colors hover:text-white">
                  See how it works
                </a>
              </div>
            </div>

            {/* A live-looking verdict card stands in for the product screenshot. */}
            <div className="rounded-3xl border-[8px] border-white/10 bg-black/60 shadow-md shadow-white/5 lg:rounded-[40px] lg:border-[10px]">
              <div className="k-streak rounded-2xl bg-white/[0.035] p-6 backdrop-blur-sm lg:rounded-[32px] lg:p-8">
                <div className="flex items-center justify-between text-xs tracking-[0.18em] text-[#b3a19f] uppercase">
                  <span>SMS &middot; scan result</span>
                  <span className="rounded-full border border-current px-2.5 py-1 text-[#ff3c1c]">
                    Malicious
                  </span>
                </div>

                <p className="mt-5 rounded-xl bg-black/50 p-4 text-sm leading-relaxed text-[#cdcdcd]">
                  &ldquo;Dear customer, your SBI KYC is incomplete. Your account will be
                  blocked today. Update now at sbi-kyc-verify.tk/update&rdquo;
                </p>

                <div className="mt-6 flex items-end gap-3">
                  <span className="font-mono text-5xl font-semibold tabular-nums text-[#ff3c1c]">
                    0.914
                  </span>
                  <span className="pb-2 text-xs text-[#b3a19f]">risk &middot; bands 0.35 / 0.70</span>
                </div>

                <div className="mt-6 space-y-3.5">
                  {BARS.map(({ label, value }) => (
                    <div key={label}>
                      <div className="flex justify-between text-xs text-[#b3a19f]">
                        <span>{label}</span>
                        <span className="font-mono tabular-nums text-white">{value.toFixed(3)}</span>
                      </div>
                      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/10">
                        <div
                          className="h-full rounded-full bg-gradient-to-l from-[#ff1e00] to-[#ff3c1c]"
                          style={{ width: (value * 100).toFixed(1) + "%" }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <p className="mt-6 text-sm leading-relaxed text-[#cdcdcd]">
                  Urgency plus an account-blocking threat, and the link resolves to a
                  free <span className="text-white">.tk</span> domain registered 3 days ago.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ---- why ---- */}
      <section id="why" className="mx-auto max-w-[1440px] px-6 pb-20 lg:pb-32 xl:px-16">
        <SectionHead
          eyebrow="WHY SENTINEL"
          title="One platform for every channel an attacker uses"
          body="Email, SMS and voice are the same attack wearing three costumes. Sentinel scores all three the same way, so the evidence is comparable across them."
        />

        <div className="relative grid grid-cols-1 lg:grid-cols-2">
          {/* the crosshair the four cells sit around */}
          <div aria-hidden className="pointer-events-none absolute inset-0 hidden lg:block">
            <div className="absolute top-0 bottom-0 left-1/2 w-px bg-gradient-to-b from-transparent via-white/15 to-transparent" />
            <div className="absolute top-1/2 right-0 left-0 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />
            <div className="absolute top-1/2 left-1/2 size-40 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#ff3c1c]/20 blur-[60px]" />
            <div className="absolute top-1/2 left-1/2 size-14 -translate-x-1/2 -translate-y-1/2 rounded-full bg-gradient-to-l from-[#ff1e00] to-[#ff3c1c]" />
          </div>

          {PILLARS.map(({ icon: Icon, title, body }) => (
            <div key={title} className="flex flex-col items-center p-6 text-center lg:px-20 lg:py-12">
              <span className="flex size-14 items-center justify-center rounded-2xl bg-white/5 ring-1 ring-white/10">
                <Icon className="size-6 text-[#ff3c1c]" />
              </span>
              <h3 className="font-heading pt-6 pb-2 text-xl md:pb-4 md:text-[32px]">{title}</h3>
              <p className="leading-loose text-[#b3a19f] lg:text-lg">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---- channels ---- */}
      <section id="channels" className="mx-auto max-w-[1440px] px-6 pb-20 lg:pb-32 xl:px-16">
        <SectionHead
          eyebrow="CHANNELS"
          title="Every layer of the attack, measured"
          body="No number on this page is borrowed from a paper. Each one was scored in this project, on sources the model never trained on."
        />

        <div className="grid gap-6 lg:grid-cols-3">
          {CHANNELS.map(({ icon: Icon, tag, title, metric, unit, body }) => (
            <div
              key={tag}
              className="k-streak rounded-2xl bg-white/[0.035] p-8 ring-1 ring-white/10 backdrop-blur-sm"
            >
              <div className="flex items-center gap-3">
                <span className="flex size-10 items-center justify-center rounded-xl bg-gradient-to-l from-[#ff1e00] to-[#ff3c1c]">
                  <Icon className="size-5" />
                </span>
                <span className="text-xs tracking-[0.18em] text-[#b3a19f] uppercase">{tag}</span>
              </div>
              <h3 className="font-heading pt-6 text-[28px] tracking-tight">{title}</h3>
              <div className="flex items-end gap-2 pt-4">
                <span className="font-mono text-4xl font-semibold tabular-nums text-[#ff3c1c]">
                  {metric}
                </span>
                <span className="pb-1.5 text-xs text-[#b3a19f]">{unit}</span>
              </div>
              <p className="pt-5 leading-relaxed text-[#b3a19f]">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---- how it works ---- */}
      <section id="how" className="mx-auto max-w-[1440px] px-6 pb-20 lg:pb-32 xl:px-16">
        <SectionHead
          eyebrow="HOW IT WORKS"
          title="From suspicious message to explained verdict"
          body="Three steps, no integration work and nothing to install."
        />

        <div className="grid gap-6 lg:grid-cols-3">
          {STEPS.map(({ n, title, body }) => (
            <div key={n} className="k-streak rounded-2xl bg-white/[0.02] p-8">
              <span className="font-heading bg-gradient-to-l from-[#ff1e00] to-[#ff3c1c] bg-clip-text text-6xl font-semibold text-transparent">
                {n}
              </span>
              <h3 className="font-heading pt-6 pb-3 text-2xl tracking-tight">{title}</h3>
              <p className="leading-relaxed text-[#b3a19f]">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ---- cta ---- */}
      <section className="mx-auto max-w-[1440px] px-6 pb-20 lg:pb-28 xl:px-16">
        <div className="k-streak relative overflow-hidden rounded-3xl bg-white/[0.035] px-8 py-16 text-center ring-1 ring-white/10 lg:px-20 lg:py-24">
          <div
            aria-hidden
            className="absolute -top-24 left-1/2 size-[420px] -translate-x-1/2 rounded-full bg-[#ff3c1c]/25 blur-[120px]"
          />
          <div className="relative">
            <h2 className="k-title mx-auto max-w-3xl text-4xl font-semibold md:text-[56px]">
              Stop the message before it costs somebody
            </h2>
            <p className="mx-auto max-w-xl pt-6 pb-10 text-[#f9e7e5] lg:text-lg">
              Create an account and run your first scan in under a minute.
            </p>
            <Button asChild size="lg" className="h-14 rounded-xl px-8 text-base">
              <Link href={cta.href}>
                {cta.label} <ArrowRight className="ml-1 size-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* ---- footer ---- */}
      <footer className="border-t border-white/10">
        <div className="mx-auto flex max-w-[1440px] flex-col gap-6 px-6 py-10 text-sm text-[#b3a19f] md:flex-row md:items-center md:justify-between xl:px-16">
          <div className="flex items-center gap-2.5">
            <span className="flex size-7 items-center justify-center rounded-lg bg-gradient-to-l from-[#ff1e00] to-[#ff3c1c]">
              <ShieldCheck className="size-4" />
            </span>
            <span className="font-heading font-semibold tracking-tight text-white">Sentinel</span>
          </div>
          <p className="max-w-lg leading-relaxed">
            Unified phishing, smishing and vishing detection: content models, link
            analysis and sender behaviour fused into one explainable verdict.
          </p>
          <p>&copy; {new Date().getFullYear()} Sentinel</p>
        </div>
      </footer>
    </div>
  );
}

function SectionHead({ eyebrow, title, body }) {
  return (
    <div className="mx-auto flex max-w-[1000px] flex-col items-center pb-8 text-center lg:pb-12">
      <h4 className="text-sm font-semibold tracking-[0.18em] text-[#b3a19f] uppercase">{eyebrow}</h4>
      <h2 className="k-title py-4 text-4xl font-medium md:py-6 md:text-[56px]">{title}</h2>
      <p className="max-w-[620px] text-base text-[#f9e7e5] lg:text-lg">{body}</p>
    </div>
  );
}
