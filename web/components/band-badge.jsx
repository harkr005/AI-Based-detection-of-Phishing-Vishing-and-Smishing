import { Badge } from "@/components/ui/badge";

/**
 * The verdict badge on a scan row — §3's three bands, plus one.
 *
 * A SAFE scan whose content label is `spam` says SPAM instead. Fusion returns
 * SAFE for junk, so without this an unsolicited promo and a genuine bank alert
 * wear the same green badge and the class is invisible. Same rule, same
 * wording and the same colour as the phone's `Verdict.kt::rowBadge` — one of
 * the two drifting is how the app and the dashboard start disagreeing about a
 * message they both scored.
 *
 * Gmail mail that Gmail itself filed under Promotions says PROMOTIONAL the
 * same way (`gmailLabels` = scans.detail.gmail_labels). That is Gmail's tab,
 * not our model's call - shown because "SAFE" on a sale mail reads as an
 * endorsement. Gmail-only; the phone has no equivalent.
 *
 * It is NOT a fourth band and cannot become one: only the spam label (or the
 * Promotions tab) qualifies, and only under SAFE. A *smishing* label under a SAFE verdict
 * deliberately gets nothing, because that pair means fusion overruled the
 * content model on the evidence — printing SCAM over a Safe verdict would
 * contradict the verdict rather than explain it.
 */
export function BandBadge({ verdict, label, gmailLabels }) {
  const spam = verdict === "SAFE" && label === "spam";
  const promo = verdict === "SAFE" && !spam && gmailLabels?.includes("CATEGORY_PROMOTIONS");
  return (
    <Badge
      variant="outline"
      className={`${spam ? "band-SPAM" : promo ? "band-PROMO" : `band-${verdict}`} border-current text-xs`}
    >
      {spam ? "SPAM" : promo ? "PROMOTIONAL" : verdict}
    </Badge>
  );
}
