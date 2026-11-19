import Link from "next/link";
import { ShieldCheck } from "lucide-react";

export function AuthShell({ title, subtitle, children }) {
  return (
    <div className="k-ambient relative flex min-h-dvh flex-col items-center justify-center px-6 py-16 text-white">
      <Link href="/" className="flex items-center gap-2.5">
        <span className="flex size-9 items-center justify-center rounded-lg bg-gradient-to-l from-[#ff1e00] to-[#ff3c1c]">
          <ShieldCheck className="size-5" />
        </span>
        <span className="font-heading text-xl font-semibold tracking-tight">Sentinel</span>
      </Link>

      <h1 className="k-title pt-8 text-center text-4xl font-semibold">{title}</h1>
      <p className="pt-3 pb-8 text-center text-[#b3a19f]">{subtitle}</p>

      {children}
    </div>
  );
}
