"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BandBadge } from "@/components/band-badge";

// One row of a feed (Phone SMS, Gmail): kept to two lines so a screen holds
// more of them. The risk number lives in View's technical details, not here.
//
// ponytail: the checkbox only tracks selection for now — what a selection DOES
// (delete, mark safe, ...) is not decided yet.
export function ScanRow({ r, text, badges, picked, onPick, onView }) {
  const sender = r.sender || "Unknown sender";
  return (
    // The checkbox sits outside the card, in the gutter to its left.
    <div className="flex items-center gap-3">
      <input
        type="checkbox"
        checked={picked}
        onChange={onPick}
        aria-label={`Select message from ${sender}`}
        className="size-4 shrink-0 cursor-pointer accent-primary"
      />
      <Card size="sm" className="min-w-0 flex-1">
        <CardContent className="flex items-center gap-3">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <p className="truncate text-sm font-semibold">{sender}</p>
              <BandBadge verdict={r.verdict} label={r.detail?.label} gmailLabels={r.detail?.gmail_labels} />
              {badges}
              <span className="ml-auto shrink-0 pl-2 text-xs tabular-nums text-muted-foreground">
                {new Date(r.created_at).toLocaleString()}
              </span>
            </div>
            {/* Capped, not just one line: on a wide screen one line holds a whole
                SMS, so nothing was ever cut. Like the phone, a glimpse ending in
                "..." - View has the rest. */}
            <p className="max-w-xl truncate text-sm text-muted-foreground">{text}</p>
          </div>
          <Button variant="outline" size="lg" className="shrink-0 px-5" onClick={onView}>
            View
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

// Selection state for a feed's checkboxes: a Set of row ids.
export const togglePick = (set, id) => {
  const next = new Set(set);
  next.has(id) ? next.delete(id) : next.add(id);
  return next;
};
