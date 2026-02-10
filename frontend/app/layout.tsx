import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "EmotionAI - Multimodal Emotion Detection",
  description: "Real-time emotion detection using video and audio analysis powered by AI",
  keywords: ["emotion detection", "AI", "machine learning", "facial recognition", "audio analysis"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className={`${inter.className} antialiased`}>
        {/* Background effects */}
        <div className="fixed inset-0 bg-gradient-cyber gradient-shift pointer-events-none z-0" />
        <div className="fixed inset-0 grid-bg pointer-events-none z-0" />

        {/* Main content */}
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  );
}
