'use client';

import { useEffect, useState } from 'react';
import { Video as VideoIcon, Music, Clock, Activity } from 'lucide-react';
import { EMOTION_ICONS, EMOTION_COLORS } from '@/lib/config';
import { formatSessionTime } from '@/lib/mediaCapture';
import type { LogEntry } from '@/types';
import { AudioProcessor } from '@/lib/audioProcessor';
import AudioVisualizer from './AudioVisualizer';

interface ResultsPanelProps {
    emotion: string;
    framesProcessed: number;
    audioChunks: number;
    sessionStartTime: number | null;
    logs: LogEntry[];
    audioProcessor: AudioProcessor | null;
}

export default function ResultsPanel({
    emotion,
    framesProcessed,
    audioChunks,
    sessionStartTime,
    logs,
    audioProcessor,
}: ResultsPanelProps) {
    const [sessionTime, setSessionTime] = useState('00:00');

    // Update session time
    useEffect(() => {
        if (!sessionStartTime) {
            setSessionTime('00:00');
            return;
        }

        const interval = setInterval(() => {
            const elapsed = Date.now() - sessionStartTime;
            setSessionTime(formatSessionTime(elapsed));
        }, 1000);

        return () => clearInterval(interval);
    }, [sessionStartTime]);

    const displayEmotion = typeof emotion === 'object' ? (emotion as any).emotion : emotion;
    const emotionLower = String(displayEmotion || '').toLowerCase();
    const emotionIcon = EMOTION_ICONS[emotionLower] || EMOTION_ICONS.default;
    const emotionColor = EMOTION_COLORS[emotionLower] || EMOTION_COLORS.default;

    return (
        <div className="glass rounded-2xl overflow-hidden flex flex-col h-full border-white/5 bg-white/5 backdrop-blur-2xl shadow-2xl transition-all duration-700">
            {/* Header / Title */}
            <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-neon-purple" />
                    <h2 className="text-xs font-black uppercase tracking-[0.2em] text-gray-400">Live Analysis</h2>
                </div>
                <div className="flex gap-1">
                    <div className="w-1.5 h-1.5 rounded-full bg-red-500/50 animate-pulse"></div>
                    <div className="w-1.5 h-1.5 rounded-full bg-yellow-500/50"></div>
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500/50"></div>
                </div>
            </div>

            <div className="p-4 space-y-4 flex-1">
                {/* Visual Emotion Display */}
                <div className="relative group overflow-hidden rounded-xl bg-black/40 border border-white/5 p-6 transition-all duration-500">
                    {/* Background Dynamic Glow */}
                    <div className={`absolute inset-0 opacity-20 bg-gradient-to-br ${emotionColor} blur-3xl group-hover:opacity-30 transition-opacity duration-1000 transform scale-150`}></div>

                    <div className="relative z-10 text-center space-y-2">
                        <div className="emotion-icon inline-block drop-shadow-[0_0_15px_rgba(255,255,255,0.3)] text-5xl">
                            {emotionIcon}
                        </div>
                        <h3 className={`text-4xl font-black uppercase tracking-tighter bg-gradient-to-b from-white to-white/50 bg-clip-text text-transparent`}>
                            {displayEmotion}
                        </h3>
                        <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Primary Emotion detected</p>
                    </div>
                </div>

                {/* Stats Grid - Ultra Compact */}
                <div className="grid grid-cols-3 gap-2">
                    <div className="bg-white/5 border border-white/5 rounded-lg p-2.5 flex flex-col items-center justify-center transition-colors hover:bg-white/10">
                        <VideoIcon className="w-3.5 h-3.5 text-neon-cyan mb-1" />
                        <span className="text-sm font-black tracking-tighter">{framesProcessed}</span>
                        <span className="text-[8px] text-gray-500 uppercase font-bold tracking-wider">Frames</span>
                    </div>

                    <div className="bg-white/5 border border-white/5 rounded-lg p-2.5 flex flex-col items-center justify-center transition-colors hover:bg-white/10">
                        <Music className="w-3.5 h-3.5 text-neon-pink mb-1" />
                        <span className="text-sm font-black tracking-tighter">{audioChunks}</span>
                        <span className="text-[8px] text-gray-500 uppercase font-bold tracking-wider">Audio</span>
                    </div>

                    <div className="bg-white/5 border border-white/5 rounded-lg p-2.5 flex flex-col items-center justify-center transition-colors hover:bg-white/10">
                        <Clock className="w-3.5 h-3.5 text-neon-purple mb-1" />
                        <span className="text-sm font-black tracking-tighter">{sessionTime}</span>
                        <span className="text-[8px] text-gray-500 uppercase font-bold tracking-wider">Time</span>
                    </div>
                </div>

                {/* Audio Visualizer Area */}
                <div className="glass rounded-lg p-3 bg-black/20 border-white/5 min-h-[60px] flex items-center justify-center overflow-hidden">
                    <AudioVisualizer audioProcessor={audioProcessor} />
                </div>

                {/* Logs - More compact and terminal-like */}
                <div className="flex flex-col flex-1 min-h-0">
                    <div className="flex items-center justify-between mb-1.5">
                        <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">System Logs</span>
                    </div>
                    <div className="bg-black/40 rounded-lg p-3 border border-white/5 grow overflow-hidden flex flex-col">
                        <div className="max-h-24 overflow-y-auto custom-scrollbar space-y-1">
                            {logs.length === 0 ? (
                                <div className="flex items-center gap-2 text-[10px] text-gray-600 font-mono italic">
                                    <span className="animate-pulse">_</span>
                                    <span>Awaiting data stream...</span>
                                </div>
                            ) : (
                                logs.map((log, index) => (
                                    <div key={index} className={`flex gap-2 text-[10px] font-mono leading-tight log-${log.level}`}>
                                        <span className="opacity-40 shrink-0">[{log.timestamp}]</span>
                                        <span className="break-words select-all">{log.message}</span>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
