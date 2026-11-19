import { redirect } from "next/navigation";

import { AuthShell } from "@/components/auth-shell";
import { AuthForm } from "@/components/auth-form";
import { signUp } from "@/app/auth-actions";
import { userId } from "@/lib/auth";

export const metadata = { title: "Create an account — Sentinel" };

export default async function SignUpPage() {
  if (await userId()) redirect("/dashboard");
  return (
    <AuthShell title="Get started" subtitle="Run your first scan in under a minute.">
      <AuthForm mode="signup" action={signUp} />
    </AuthShell>
  );
}
