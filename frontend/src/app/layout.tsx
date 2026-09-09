import type { Metadata } from "next";
import { JetBrains_Mono, Space_Grotesk } from "next/font/google";

import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-grotesk",
  display: "swap",
});
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  title: "NeuroNest — retrieval-augmented generation, measured",
  description:
    "A question answering service that indexes your documents and answers strictly from the passages it retrieved, declining when the corpus does not contain the answer.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={[spaceGrotesk.variable, jetbrainsMono.variable, "antialiased"].join(" ")}>
      <body>{children}</body>
    </html>
  );
}
