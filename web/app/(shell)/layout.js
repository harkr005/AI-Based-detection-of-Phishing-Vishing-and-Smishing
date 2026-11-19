import { redirect } from "next/navigation";

import { AppSidebar } from "@/components/app-sidebar";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { currentUser } from "@/lib/auth";

// Everything under (shell) is the signed-in console. This redirect is the gate
// — there is no middleware to keep in sync with it, and the API routes gate
// themselves on the same cookie via userId().
export default async function ShellLayout({ children }) {
  const user = await currentUser();
  if (!user) redirect("/sign-in");

  return (
    <SidebarProvider className="k-ambient">
      <AppSidebar user={user.username} />
      <SidebarInset className="bg-transparent">
        {/* No header bar: sign-out lives in the sidebar's user menu. On a phone
            the sidebar is off-canvas, so this button is the only way to open
            it; on desktop the sidebar's rail (and Ctrl+B) collapses it. */}
        <div className="px-4 pt-4 md:hidden">
          <SidebarTrigger />
        </div>
        <main className="flex-1 p-4 md:p-8">{children}</main>
      </SidebarInset>
    </SidebarProvider>
  );
}
