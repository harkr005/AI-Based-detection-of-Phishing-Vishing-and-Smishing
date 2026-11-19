"use client";

import { useActionState, useState } from "react";
import Link from "next/link";
import { AlertCircle, ArrowRight, Check, Eye, EyeOff, Loader2, X } from "lucide-react";

import { RULES } from "@/lib/password";

export function AuthForm({ mode, action }) {
  const isSignUp = mode === "signup";
  const [state, formAction, pending] = useActionState(action, null);
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);

  return (
    <div className="k-streak w-full max-w-[440px] rounded-2xl bg-white/[0.04] p-8 ring-1 ring-white/10 backdrop-blur-sm">
      <h2 className="font-heading text-center text-2xl font-semibold tracking-tight">
        {isSignUp ? "Create your account" : "Sign in to Sentinel"}
      </h2>
      <p className="pt-2 pb-7 text-center text-sm text-[#b3a19f]">
        {isSignUp
          ? "Pick a username and a password. Nothing else is asked for."
          : "Welcome back. Enter your credentials to continue."}
      </p>

      <form action={formAction} className="space-y-5">
        <div className="space-y-2">
          <label htmlFor="username" className="block text-sm font-medium">
            Username
          </label>
          <input
            id="username"
            name="username"
            autoComplete="username"
            required
            placeholder="Enter your username"
            className="h-12 w-full rounded-xl border border-white/10 bg-black/40 px-4 text-sm outline-none transition-colors placeholder:text-[#6d5b59] focus:border-[#ff3c1c]/60 focus:ring-2 focus:ring-[#ff3c1c]/25"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="password" className="block text-sm font-medium">
            Password
          </label>
          <div className="relative">
            <input
              id="password"
              name="password"
              type={show ? "text" : "password"}
              autoComplete={isSignUp ? "new-password" : "current-password"}
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={isSignUp ? "Create a strong password" : "Enter your password"}
              className="h-12 w-full rounded-xl border border-white/10 bg-black/40 px-4 pr-11 text-sm outline-none transition-colors placeholder:text-[#6d5b59] focus:border-[#ff3c1c]/60 focus:ring-2 focus:ring-[#ff3c1c]/25"
            />
            <button
              type="button"
              onClick={() => setShow((v) => !v)}
              aria-label={show ? "Hide password" : "Show password"}
              className="absolute top-0 right-0 flex h-12 w-11 items-center justify-center text-[#b3a19f] transition-colors hover:text-white"
            >
              {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
            </button>
          </div>
        </div>

        {isSignUp && (
          <ul className="space-y-1.5 rounded-xl bg-black/30 p-3.5">
            {RULES.map(({ label, ok }) => {
              const passed = ok(password);
              return (
                <li
                  key={label}
                  className={`flex items-center gap-2 text-xs transition-colors ${
                    passed ? "text-[#4ade80]" : "text-[#b3a19f]"
                  }`}
                >
                  {passed ? <Check className="size-3.5" /> : <X className="size-3.5 opacity-50" />}
                  {label}
                </li>
              );
            })}
          </ul>
        )}

        {state?.error && (
          <p className="flex items-start gap-2 rounded-xl border border-[#ff3c1c]/40 bg-[#ff3c1c]/10 p-3 text-sm text-[#ff9c8a]">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            {state.error}
          </p>
        )}

        <button
          type="submit"
          disabled={pending}
          className="k-glow flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-l from-[#ff1e00] to-[#ff3c1c] font-medium transition-all duration-300 disabled:opacity-60"
        >
          {pending ? (
            <>
              <Loader2 className="size-4 animate-spin" /> Just a second
            </>
          ) : (
            <>
              {isSignUp ? "Create account" : "Continue"} <ArrowRight className="size-4" />
            </>
          )}
        </button>
      </form>

      <p className="pt-7 text-center text-sm text-[#b3a19f]">
        {isSignUp ? "Already have an account? " : "Don't have an account? "}
        <Link
          href={isSignUp ? "/sign-in" : "/sign-up"}
          className="font-medium text-[#ff3c1c] transition-opacity hover:opacity-80"
        >
          {isSignUp ? "Sign in" : "Sign up"}
        </Link>
      </p>
    </div>
  );
}
