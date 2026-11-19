"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ChevronsUpDown, Inbox, LayoutDashboard, LogOut, MessageSquare, Mail, PanelLeftClose,
  PanelLeftOpen, PhoneCall, Settings, ShieldCheck, Smartphone,
} from "lucide-react";

import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupContent,
  SidebarGroupLabel, SidebarHeader, SidebarMenu, SidebarMenuButton,
  SidebarMenuItem, SidebarRail, useSidebar,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel,
  DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { signOut } from "@/app/auth-actions";

const NAV = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/sms", label: "SMS Check", icon: MessageSquare },
  { href: "/email", label: "Email Check", icon: Mail },
  { href: "/voice", label: "Voice Check", icon: PhoneCall },
  { href: "/phone", label: "Phone App", icon: Smartphone },
  { href: "/gmail", label: "Gmail", icon: Inbox },
];

export function AppSidebar({ user }) {
  const path = usePathname();
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b border-sidebar-border">
        <div className="flex items-center">
          {/* Collapsed to icons there is no room for the name: only the button
              that opens it again stays. */}
          <Link href="/" className="flex min-w-0 items-center gap-2 px-2 py-1.5 group-data-[collapsible=icon]:hidden">
            <div className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-l from-[#ff1e00] to-[#ff3c1c]">
              <ShieldCheck className="size-4 text-primary-foreground" />
            </div>
            <span className="truncate font-heading text-base font-semibold tracking-tight group-data-[collapsible=icon]:hidden">
              SENTINEL
            </span>
          </Link>
          <CollapseButton />
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Detection</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV.map(({ href, label, icon: Icon }) => (
                <SidebarMenuItem key={href}>
                  <SidebarMenuButton
                    asChild
                    isActive={path === href}
                    tooltip={label}
                    className="rounded-lg font-heading data-[active=true]:bg-gradient-to-l data-[active=true]:from-[#ff1e00] data-[active=true]:to-[#ff3c1c] data-[active=true]:text-primary-foreground"
                  >
                    <Link href={href}>
                      <Icon />
                      <span>{label}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-sidebar-border">
        <UserMenu user={user} />
      </SidebarFooter>
      {/* The edge strip that collapses the sidebar, now the header bar and its
          trigger are gone. Ctrl+B does the same. */}
      <SidebarRail />
    </Sidebar>
  );
}

// Roomier than the menu default: these are the account's only two actions.
const ITEM = "gap-2.5 px-2.5 py-2.5 [&_svg:not([class*='size-'])]:size-4.5";

// Right end of the header. An icon, not a word: it closes the sidebar (or
// shuts it on a phone, where it slides over the page) and, collapsed, opens it.
function CollapseButton() {
  const { state, isMobile, toggleSidebar } = useSidebar();
  const closing = isMobile || state === "expanded";
  const Icon = closing ? PanelLeftClose : PanelLeftOpen;
  const label = closing ? "Close sidebar" : "Open sidebar";
  return (
    <Button variant="ghost" size="icon" onClick={toggleSidebar} aria-label={label} title={label}
            className="ml-auto shrink-0 text-muted-foreground hover:text-foreground group-data-[collapsible=icon]:mx-auto">
      <Icon className="size-4.5" />
    </Button>
  );
}

// The whole row is the trigger; the menu opens upward from it. Sign out posts
// the same server action the old header button did.
function UserMenu({ user }) {
  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <SidebarMenuButton size="lg" tooltip={user}
                               className="data-[state=open]:bg-sidebar-accent">
              <Avatar className="size-8 rounded-lg">
                <AvatarFallback className="rounded-lg bg-gradient-to-l from-[#ff1e00] to-[#ff3c1c] font-semibold text-primary-foreground">
                  {user.slice(0, 1).toUpperCase()}
                </AvatarFallback>
              </Avatar>
              <span className="truncate font-medium">{user}</span>
              <ChevronsUpDown className="ml-auto size-4 opacity-60" />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="top" align="start"
                               className="w-(--radix-dropdown-menu-trigger-width) min-w-60 p-1.5">
            <DropdownMenuLabel className="truncate px-2.5 py-2 text-sm font-normal text-muted-foreground">
              Signed in as <span className="font-medium text-foreground">{user}</span>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            {/* ponytail: disabled until the settings page exists — a link now
                would only lead to a 404. */}
            <DropdownMenuItem disabled className={ITEM}>
              <Settings /> Settings
              <span className="ml-auto text-xs opacity-60">soon</span>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            {/* Called directly, not through a <form>: choosing an item closes
                the menu, which can unmount a form before it submits. */}
            <DropdownMenuItem variant="destructive" className={ITEM} onSelect={() => signOut()}>
              <LogOut /> Sign out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
