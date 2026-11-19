"use client";

import { useState } from "react";
import {
  ShieldCheck, ShieldAlert, ShieldX, Flag, Loader2,
  TriangleAlert, CircleCheck, CircleHelp, Lightbulb,
} from "lucide-react";
import { toast } from "sonner";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

// The phone's own format (Verdict.kt percent): one decimal, ".0" dropped.
export const pct = (v) => `${((v ?? 0) * 100).toFixed(1).replace(/\.0$/, "")}%`;
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// §3's three bands, and nothing else is ever rendered as a verdict. `say` is the
// same three bands in the words someone who has never heard of phishing scoring
// would use — the numbers moved under "Technical details".
const BAND = {
  SAFE:       { label: "Safe",       say: "Nothing dangerous found",
                Icon: ShieldCheck, ring: "border-emerald-500/40 bg-emerald-500/10" },
  SUSPICIOUS: { label: "Suspicious", say: "Be careful with this one",
                Icon: ShieldAlert, ring: "border-amber-500/40 bg-amber-500/10" },
  PHISHING:   { label: "Malicious",  say: "This looks like a scam",
                Icon: ShieldX,     ring: "border-primary/50 bg-primary/10" },
};

const KIND = {
  ham:      "It reads like an ordinary, genuine message.",
  spam:     "It reads like unwanted marketing — junk mail rather than a scam.",
  phishing: "It reads like phishing: mail that pretends to be someone you trust so you hand over a password, a card number or money.",
  smishing: "It reads like a scam text: it pretends to be your bank, a delivery service or a government office to get your details or your money.",
};

const PRESSURE = {
  authority: "speaks as if it were a bank, the government or an official body",
  urgency:   "pushes you to act immediately",
  scarcity:  "says the offer is about to run out",
  fear:      "threatens that something bad will happen to you",
  reward:    "promises a prize, a refund or free money",
};

// url_model's own feature names (url_model.PHRASE) said plainly. Anything not
// listed falls through unchanged, so a new feature degrades to its short name
// rather than disappearing.
const LINK_SAY = {
  "rare top-level domain": "its ending (.tk, .top, .xyz and the like) is one scammers buy because it is cheap",
  "unusual TLD length": "the part after the last dot is an odd length for a real site",
  "hyphen-stuffed host": "the address is padded with hyphens to look like a real brand",
  "raw IP address instead of a domain": "it is a bare numeric address, not a company's domain",
  "punycode (look-alike) domain": "it uses look-alike foreign characters to imitate a real name",
  "bank/payment lure words in the host": "the address itself contains bank or payment words to look official",
  "digits in the host": "there are digits inside the address a real brand would not use",
  "host is mostly digits": "the address is mostly numbers",
  "deep subdomain nesting": "the real domain is hidden behind extra sub-parts",
  "unusually long host name": "the address is unusually long",
  "one very long host token": "one chunk of the address is unusually long",
  "many dot-separated parts": "the address is split into unusually many parts",
  "odd characters in the host": "there are characters in it a normal address would not have",
  "random-looking host string": "the address looks randomly generated",
  "host reads as non-word letters": "the address is not made of real words",
};

// What each fusion channel is, said the way the headline needs it.
const DRIVER = {
  nlp: (r) => (r.channel === "voice" ? "what the caller said" : "the wording of the message"),
  url: (r) => (r.attachments?.bad?.length ? "the attached file" : "the link in it"),
  behaviour: () => "who sent it",
  spoof: () => "the voice itself",
};

/** Each channel's share of the fused risk: fusion.fuse's own weighted sum,
 *  weight x score, so the ranking below is the arithmetic that set the band. */
function shares(r) {
  const w = r.weights ?? {};
  return Object.fromEntries(Object.entries(r.parts ?? {}).map(([k, v]) => [k, (w[k] ?? 0) * v]));
}

/** One sentence on what decided the verdict. The three override rules come
 *  first because when one fired, the weighted sum is not what set the band. */
function headline(r) {
  if (r.deescalated)
    return "It reads like a scam, but it comes from a registered sender, has no bad link and asks for nothing, so it was cleared.";
  if (r.escalated)
    return "The message itself looks ordinary, but the sender does not behave like anyone you normally hear from, so it is held as suspicious.";
  if (r.routed)
    return "The checks could not make up their mind about this one, so it is held as suspicious rather than cleared.";
  if (r.verdict === "SAFE") return null;
  const mv = r.moves ?? [];
  if (mv.length)
    return `Flagged because the call contains ${mv[0].say}${mv.length > 1 ? `, plus ${mv.length - 1} more scam sign${mv.length > 2 ? "s" : ""} listed below` : ""}.`;
  const s = shares(r);
  const top = Object.keys(s).sort((a, b) => s[b] - s[a])[0];
  if (!top || !DRIVER[top]) return null;
  // Say so when the text model disagreed with whatever carried the verdict -
  // that is the case a reader otherwise finds contradictory.
  const cleanText = top !== "nlp" && r.parts?.nlp != null && r.parts.nlp < 0.35;
  return `Flagged mainly because of ${DRIVER[top](r)}${cleanText ? " — the wording on its own looked normal" : ""}.`;
}

/** The whole plain-language layer: structured fields -> sentences. Nothing here
 *  re-decides anything, it only says out loud what the verdict was built from.
 *  Each point names its fusion channel so the list can lead with what counted. */
function plainPoints(r) {
  const out = [];
  const worst = r.links?.[0];

  if (r.solicitation?.length)
    out.push({ ch: "nlp", bad: true, text: "It asks you to pay to sign up and pushes a referral code — the shape of paid internship and certificate offers that arrive uninvited." });

  // Gmail only (attachments.py). The type, never the filename: scans are stored.
  for (const b of r.attachments?.bad ?? [])
    out.push({ ch: "url", bad: true, text: `The attached ${b.type} file is ${b.why}. Do not open it.` });
  if (r.attachments?.count && !r.attachments.bad?.length)
    out.push({ ch: "url", bad: false, text: r.attachments.count > 1
      ? `None of its ${r.attachments.count} attached files is a type that can run a program.`
      : "Its attached file is not a type that can run a program." });

  if (r.parts?.spoof != null)
    out.push(r.parts.spoof >= 0.5
      ? { ch: "spoof", bad: true, text: "The voice sounds artificially generated rather than a real person." }
      : { ch: "spoof", bad: false, text: "No clear sign of a synthetic voice. Most scam calls are real people, so this does not lower the risk." });

  // Voice: each scam move, when it was heard and the sentence it was in
  // (voice_model.moves). Older saved scans have no `moves` and show neither line.
  // "The call", not "the caller": a recording holds both voices.
  for (const m of r.moves ?? [])
    out.push({ ch: "nlp", bad: true, text: `At ${clock(m.t)}: ${m.say}.`, quote: m.quote });
  if (r.moves?.length === 0)
    out.push({ ch: "nlp", bad: false, text: "Nobody on the call asked for a code, money or access to your phone, and there were no threats." });

  // The exact words the content model leaned on (sms_model.explain returns them
  // as data). This is the question a new user actually asks — "what did it see?"
  if (r.terms?.length)
    out.push({
      ch: "nlp",
      bad: r.label !== "ham",
      terms: r.terms,
      text: r.label === "ham"
        ? "These everyday words are what made it read as an ordinary message:"
        : "These words in it carried the most weight:",
    });

  if (worst) {
    // ...and WHY the link is wrong, in words, not feature names. The address
    // and the reasons stay separate fields so each gets its own line.
    const why = (worst.why ?? []).map((w) => LINK_SAY[w] ?? w);
    if (worst.attachment) why.unshift(`This link came from the attached ${worst.attachment}, not the email text.`);
    // Google's own listing outranks the model's guesses. "when it was scanned":
    // the Safe Browsing terms allow a warning only on data under 30 minutes old,
    // and a stored scan is read later. The attribution line is required.
    if (worst.listed)
      out.push({ ch: "url", bad: true, url: worst.url,
                 why: ["Advisory provided by Google. Google's list may not be complete or accurate."],
                 text: `Google Safe Browsing listed this link as ${worst.listed} when it was scanned. Do not open it.` });
    else out.push(worst.score >= 0.7
      ? { ch: "url", bad: true, url: worst.url, why, text: "The link does not look like a real company's website." }
      : worst.score >= 0.35
        ? { ch: "url", bad: true, url: worst.url, why, text: "The link looks a little odd — check it before you tap it." }
        : { ch: "url", bad: false, text: "The link in it looks like a normal website address." });
  }
  else if (r.channel !== "voice")
    out.push({ bad: false, text: "There is no link in it to tap." });

  const hits = Object.entries(r.persuasion ?? {})
    .filter(([, v]) => v >= 0.5).map(([k]) => PRESSURE[k]);
  if (hits.length)
    out.push({ ch: "nlp", bad: true, text: `It ${hits.join(", and ")} — the pressure tactics scams rely on.` });

  if (r.parts?.behaviour != null)
    out.push(r.parts.behaviour >= 0.5
      ? { ch: "behaviour", bad: true, text: "The sender does not behave like the people you normally hear from: new address, unusual hour or unusual wording." }
      : { ch: "behaviour", bad: false, text: "The sender behaves like someone you normally hear from." });
  else if (r.channel !== "voice")
    out.push({ note: true, text: "The sender could not be judged yet — this app has not seen enough of your past messages to know who is normal for you." });

  if (r.evidence === 0)
    out.push({ ch: "nlp", bad: false, text: "It does not actually ask you to do anything: no code to share, no payment, no threat." });

  // Warnings first, biggest share of the risk first; then reassurances, then
  // notes. Array.sort is stable, so ties keep the order written above.
  const s = shares(r);
  const rank = (p) => (p.bad ? 0 : p.note ? 2 : 1);
  return out.sort((a, b) => rank(a) - rank(b) || (s[b.ch] ?? 0) - (s[a.ch] ?? 0));
}

// What to actually do, by band. Advice only: it restates the verdict as an
// action and never re-decides anything.
const ADVICE = {
  SAFE: "Nothing to do. If it ever asks for a code, a password or money, scan it again.",
  SUSPICIOUS: "Don't tap links or reply yet. If it claims to be a company you use, check through their official app or website instead.",
  PHISHING: "Don't tap anything, reply or share any code. Delete it. If it claims to be your bank, call the number on the back of your card.",
};

// One group of findings: a small heading and its points, each on its own line.
const TONE = {
  bad:  { Icon: TriangleAlert, cls: "band-PHISHING" },
  good: { Icon: CircleCheck,   cls: "band-SAFE" },
  note: { Icon: CircleHelp,    cls: "text-muted-foreground" },
};

function Findings({ title, items, tone }) {
  if (!items.length) return null;
  const { Icon, cls } = TONE[tone];
  return (
    <section className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</h3>
      <ul className="space-y-3">
        {items.map((p, i) => (
          <li key={i} className="flex gap-2.5 text-sm leading-relaxed">
            <Icon className={`mt-0.5 size-4 shrink-0 ${cls}`} />
            <div className={`min-w-0 space-y-1.5 ${tone === "note" ? "text-muted-foreground" : ""}`}>
              <p>{p.text}</p>
              {p.quote && (
                <blockquote className="border-l-2 pl-3 text-sm italic text-muted-foreground">“{p.quote}”</blockquote>
              )}
              {p.url && (
                <code className="block rounded bg-muted px-2 py-1 text-xs [overflow-wrap:anywhere]">{p.url}</code>
              )}
              {p.why?.length > 0 && (
                <ul className="list-disc space-y-0.5 pl-4 text-xs text-muted-foreground">
                  {p.why.map((w) => <li key={w}>{w}</li>)}
                </ul>
              )}
              {p.terms && (
                <div className="flex flex-wrap gap-1.5">
                  {p.terms.map((t) => (
                    <Badge key={t} variant="outline" className="font-mono font-normal">{t}</Badge>
                  ))}
                </div>
              )}
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Bar({ label, value, weight }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-muted-foreground">
          {label}
          {weight != null && (
            <span className="ml-1.5 opacity-60">weight {pct(weight)}</span>
          )}
        </span>
        <span className="font-mono tabular-nums">{pct(value)}</span>
      </div>
      <Progress value={(value ?? 0) * 100} className="h-1.5" />
    </div>
  );
}

/** The user correcting or confirming us: spam, scam, or (on a message we
 *  accused) not a threat. Saved on the scan's own row (scans.feedback). Only
 *  'is_a_threat' (scam) is a crowd report on a Gmail sender; spam is unwanted,
 *  not fraud, so it never moves anyone's score. Only when the scan reached
 *  Postgres, because the row is what gets marked. */
const REPORTS = {
  is_spam: { label: "Spam", done: "Recorded as spam." },
  is_a_threat: { label: "Scam", done: "Recorded as a scam." },
  not_a_threat: { label: "Not spam or a scam", done: "Recorded as a false alarm." },
};

function ReportButtons({ result }) {
  const [sent, setSent] = useState(null);  // the kind that was saved
  const [busy, setBusy] = useState(null);
  const accused = result.verdict !== "SAFE"
    || ["spam", "smishing", "phishing"].includes(result.label);
  if (!result.id) return null;
  const kinds = ["is_spam", "is_a_threat", ...(accused ? ["not_a_threat"] : [])];

  async function report(feedback) {
    setBusy(feedback);
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: result.id, feedback }),
      });
      if (!res.ok) throw new Error((await res.json()).error || "Could not send that");
      setSent(feedback);
      toast.success(`Thanks — ${REPORTS[feedback].done}`);
    } catch (e) {
      toast.error(e.message);
    } finally {
      setBusy(null);
    }
  }

  // Not locked after one click: a wrong pick can be changed, the row keeps the last.
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs text-muted-foreground">Did we get it wrong? It&apos;s really</span>
      {kinds.map((k) => (
        <Button key={k} variant={sent === k ? "default" : "outline"} size="sm"
                disabled={busy !== null || sent === k} onClick={() => report(k)}>
          {busy === k ? <Loader2 className="animate-spin" /> : <Flag />}
          {REPORTS[k].label}
        </Button>
      ))}
    </div>
  );
}

export function VerdictPanel({ result }) {
  if (!result) return null;
  const band = BAND[result.verdict] ?? BAND.SAFE;
  const { parts = {}, weights = {}, classes, persuasion, links = [] } = result;
  const points = plainPoints(result);
  const lead = headline(result);

  return (
    <div className="space-y-4">
      {/* Read top to bottom like someone explaining it: the verdict, why in
          one sentence, what we noticed, what looked fine, what to do. The
          class label (ham/spam/...) is said in KIND's words, not as a badge. */}
      <Card className={`border ${band.ring}`}>
        <CardHeader className="pb-1">
          <CardTitle className="flex items-center gap-2 text-lg">
            <band.Icon className={`size-6 band-${result.verdict}`} />
            {band.say}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-1.5 text-sm leading-relaxed">
            {lead && <p className="font-medium">{lead}</p>}
            {KIND[result.label] && <p className="text-muted-foreground">{KIND[result.label]}</p>}
          </div>

          <Findings title="What we noticed" tone="bad" items={points.filter((p) => p.bad)} />
          <Findings title="What looks fine" tone="good" items={points.filter((p) => !p.bad && !p.note)} />
          <Findings title="What we couldn't check" tone="note" items={points.filter((p) => p.note)} />

          <section className="flex gap-2.5 rounded-lg bg-muted/50 p-3 text-sm leading-relaxed">
            <Lightbulb className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
            <p><span className="font-medium">What should you do? </span>{ADVICE[result.verdict] ?? ADVICE.SAFE}</p>
          </section>

          <div className="border-t pt-4">
            <ReportButtons result={result} />
          </div>
        </CardContent>
      </Card>

      {/* Everything a marker would ask for, one click away and out of everyone
          else's way. <details> is the native disclosure widget — no state. */}
      <details className="group rounded-lg border">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium text-muted-foreground hover:text-foreground">
          Technical details
          <span className="ml-2 text-xs font-normal opacity-60 group-open:hidden">
            risk score, channel weights, class probabilities
          </span>
        </summary>

        <div className="space-y-4 border-t p-4">
          <div>
            <div className="flex items-end gap-3">
              <span className={`font-mono text-4xl font-semibold tabular-nums band-${result.verdict}`}>
                {pct(result.risk)}
              </span>
              <span className="pb-1.5 text-xs text-muted-foreground">
                risk · bands 35% / 70% · verdict {band.label}
              </span>
            </div>
            {/* No reason string here: the plain-language list above already
                says why, and repeating it as one long sentence only buried it. */}
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">
                Channel contributions
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  weights renormalise over the channels that reported
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {Object.entries(parts).map(([k, v]) => (
                <Bar
                  key={k}
                  value={v}
                  weight={weights[k]}
                  label={{ nlp: "Content (NLP)", url: "Link (XGBoost)",
                           behaviour: "Sender behaviour", spoof: "Synthetic voice" }[k] ?? k}
                />
              ))}
              {result.confidence != null && (
                <p className="pt-1 text-xs text-muted-foreground">
                  {/* entropy stays a plain number: it is an information
                      measure, not a share (the phone does the same) */}
                  model confidence {pct(result.confidence)} · entropy{" "}
                  {result.entropy?.toFixed(3)}
                </p>
              )}
            </CardContent>
          </Card>

          {classes && (
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Class probabilities</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                {Object.entries(classes).map(([k, v]) => <Bar key={k} label={k} value={v} />)}
              </CardContent>
            </Card>
          )}

          {persuasion && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">
                  Persuasion principles
                  <span className="ml-2 text-xs font-normal text-muted-foreground">
                    auxiliary head
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-wrap gap-2">
                {Object.entries(persuasion).map(([k, v]) => (
                  <Badge key={k} variant={v >= 0.5 ? "default" : "outline"} className="font-normal">
                    {k} {pct(v)}
                  </Badge>
                ))}
              </CardContent>
            </Card>
          )}

          {links.length > 0 && (
            <Card>
              <CardHeader className="pb-2"><CardTitle className="text-sm">Links found</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                {links.map((l, i) => (
                  // index too: scans stored before api._links deduped can repeat a URL
                  <div key={`${i}:${l.url}`}className="flex items-center justify-between gap-3 text-sm">
                    <code className="truncate text-xs">{l.url}</code>
                    <span className={`shrink-0 font-mono tabular-nums ${l.score >= 0.5 ? "band-PHISHING" : "text-muted-foreground"}`}>
                      {/* the Safe Browsing terms want the attribution wherever its call shows */}
                      {l.attachment && (
                        <span className="mr-2 font-sans text-xs text-muted-foreground">
                          in attached {l.attachment}
                        </span>
                      )}
                      {l.listed && (
                        <span className="mr-2 font-sans text-xs text-muted-foreground">
                          listed by Google Safe Browsing
                        </span>
                      )}
                      {pct(l.score)}
                    </span>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      </details>
    </div>
  );
}
