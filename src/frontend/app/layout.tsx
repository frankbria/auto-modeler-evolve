import type { Metadata } from "next"
import { Nunito_Sans } from "next/font/google"
import Link from "next/link"
import "./globals.css"

const nunitoSans = Nunito_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "AutoModeler",
  description: "AI-powered conversational data modeling",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`${nunitoSans.variable} font-sans antialiased`}>
        <nav className="sticky top-0 z-50 flex h-12 items-center border-b bg-background px-6">
          <Link href="/" className="text-base font-semibold tracking-tight">
            AutoModeler
          </Link>
          <div className="ml-6">
            <Link
              href="/"
              className="text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              Home
            </Link>
          </div>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  )
}
