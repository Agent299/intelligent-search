import type { Metadata } from "next";
import Link from "next/link";
import { SessionCookie } from "@/app/components/SessionCookie";
import "./globals.css";

export const metadata: Metadata = {
  title: "Intelligent Search",
  description: "Storefront demo for the AI-powered e-commerce search case study",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>
        <SessionCookie />
        <header className="site-header">
          <Link href="/" className="logo">Intelligent Search</Link>
          <span className="tagline">case-study storefront demo</span>
        </header>
        {children}
      </body>
    </html>
  );
}
