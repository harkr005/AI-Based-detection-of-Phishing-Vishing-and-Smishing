import { Manrope, Geist_Mono } from "next/font/google";
import "./globals.css";

import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";

const manrope = Manrope({ variable: "--font-manrope", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata = {
  title: "Sentinel — phishing, smishing & vishing detection",
  description: "Unified social-engineering detection across email, SMS and voice.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`dark ${manrope.variable} ${geistMono.variable} h-full antialiased`}>
      <head>
        {/* Clash Display is Fontshare-only, so it comes in as a stylesheet
            rather than through next/font. */}
        <link rel="preconnect" href="https://api.fontshare.com" />
        <link
          rel="stylesheet"
          href="https://api.fontshare.com/v2/css?f%5B%5D=clash-display@400,500,600,700&display=swap"
        />
      </head>
      <body className="min-h-full bg-background">
        <TooltipProvider>{children}</TooltipProvider>
        <Toaster position="top-right" />
      </body>
    </html>
  );
}
