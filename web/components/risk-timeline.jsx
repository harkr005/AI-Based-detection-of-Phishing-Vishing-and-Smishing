"use client";

import {
  Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

/** §6's rolling risk curve. The shape is the signal — scam calls escalate — so
 *  the §3 bands are drawn as fixed guides rather than autoscaled away. */
export function RiskTimeline({ timeline = [], alertAt, peak }) {
  if (!timeline.length) return null;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Risk over the call</CardTitle>
        <CardDescription className="text-xs">
          Peak {peak?.toFixed(3)}
          {alertAt != null
            ? ` · alert raised at ${alertAt.toFixed(1)}s`
            : " · never held above 0.70 for two consecutive windows"}
        </CardDescription>
      </CardHeader>
      <CardContent className="pl-0">
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={timeline} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
            <defs>
              <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--primary)" stopOpacity={0.45} />
                <stop offset="100%" stopColor="var(--primary)" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis
              dataKey="t" tickLine={false} axisLine={false}
              tick={{ fontSize: 11, fill: "var(--muted-foreground)" }}
              tickFormatter={(t) => `${t}s`}
            />
            <YAxis
              domain={[0, 1]} ticks={[0, 0.35, 0.7, 1]} tickLine={false} axisLine={false}
              tick={{ fontSize: 11, fill: "var(--muted-foreground)" }} width={38}
            />
            <Tooltip
              contentStyle={{
                background: "var(--popover)", border: "1px solid var(--border)",
                borderRadius: 8, fontSize: 12,
              }}
              formatter={(v) => [v.toFixed(3), "risk"]}
              labelFormatter={(t) => `${t}s`}
            />
            <ReferenceLine y={0.35} stroke="oklch(0.80 0.16 85)" strokeDasharray="4 4" />
            <ReferenceLine y={0.7} stroke="oklch(0.65 0.21 27.5)" strokeDasharray="4 4" />
            {alertAt != null && (
              <ReferenceLine x={alertAt} stroke="var(--primary)"
                label={{ value: "alert", fontSize: 10, fill: "var(--primary)", position: "top" }} />
            )}
            <Area type="monotone" dataKey="risk" stroke="var(--primary)" strokeWidth={2}
                  fill="url(#riskFill)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
