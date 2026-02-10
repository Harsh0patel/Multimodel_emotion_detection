'use client';

import { useState } from 'react';
import { Video, Upload } from 'lucide-react';
import StreamMode from '@/components/StreamMode';
import UploadMode from '@/components/UploadMode';
import Header from '@/components/Header';
import type { AppMode } from '@/types';

export default function Home() {
  const [mode, setMode] = useState<AppMode>('stream');

  return (
    <main className="min-h-screen p-3 md:p-4 bg-cyber-bg selection:bg-neon-purple/30">
      <div className="max-w-[1600px] mx-auto space-y-3">
        {/* Header */}
        <Header />

        {/* Mode Selector - Ultra Compact Pill Switcher */}
        <div className="flex justify-center">
          <div className="glass p-1 rounded-full flex gap-1 w-full max-w-[400px] bg-white/5 border-white/5 shadow-glow-sm">
            <button
              onClick={() => setMode('stream')}
              className={`flex-1 flex items-center justify-center gap-2 py-1.5 px-4 rounded-full text-xs font-bold tracking-wider uppercase transition-all duration-300 ${mode === 'stream'
                ? 'bg-gradient-primary text-white shadow-glow-sm'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
            >
              <Video className="w-3.5 h-3.5" />
              <span>Live detection</span>
            </button>

            <button
              onClick={() => setMode('upload')}
              className={`flex-1 flex items-center justify-center gap-2 py-1.5 px-4 rounded-full text-xs font-bold tracking-wider uppercase transition-all duration-300 ${mode === 'upload'
                ? 'bg-gradient-primary text-white shadow-glow-sm'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Upload Video</span>
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div className="animate-fade-in">
          {mode === 'stream' ? <StreamMode /> : <UploadMode />}
        </div>
      </div>
    </main>
  );
}
