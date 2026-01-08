import EmotionDashboard from "@/components/EmotionDashboard";
import { BrainCircuit } from "lucide-react";

export default function Home() {
  return (
    <main className="min-h-screen py-12 px-4 md:px-8">
      {/* Header */}
      <header className="max-w-6xl mx-auto mb-16 flex flex-col items-center text-center">
        <div className="flex items-center space-x-4 mb-6">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center shadow-xl shadow-primary/20">
            <BrainCircuit className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl md:text-5xl font-black tracking-tighter">
            <span className="text-gradient">EMO</span>REASON
          </h1>
        </div>
        <p className="text-xl text-white/50 max-w-2xl font-medium">
          Multimodal neural fusion system for real-time human emotion detection using
          <span className="text-white mx-1">advanced deep learning</span> pipelines.
        </p>
      </header>

      {/* Main Content */}
      <EmotionDashboard />

      {/* Footer */}
      <footer className="max-w-6xl mx-auto mt-20 pt-8 border-t border-white/5 text-center text-white/20 text-sm">
        <p>&copy; 2026 Multimodal Emotion Detection System. Powered by Fusion Models.</p>
      </footer>
    </main>
  );
}
