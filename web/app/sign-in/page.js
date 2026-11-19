import { redirect } from "next/navigation";

import { AuthShell } from "@/components/auth-shell";
import { AuthForm } from "@/components/auth-form";
import { signIn } from "@/app/auth-actions";
import { userId } from "@/lib/auth";

export const metadata = { title: "Sign in — Sentinel" };

export default async function SignInPage() {
  if (await userId()) redirect("/dashboard");
  return (
    <AuthShell title="Welcome back" subtitle="Sign in to reach your detection console.">
      <AuthForm mode="signin" action={signIn} />
    </AuthShell>
  );
}
