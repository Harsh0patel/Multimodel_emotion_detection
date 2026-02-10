'use client';

import { useState, useEffect } from 'react';
import { Brain } from 'lucide-react';
import { getWebSocketManager } from '@/lib/websocket';

export default function Header() {
    const [isConnected, setIsConnected] = useState(false);
    const wsManager = getWebSocketManager();

    useEffect(() => {
        // Initial state
        setIsConnected(wsManager.isConnected());

        // Subscribe to changes
        const unsubscribe = wsManager.onStatusChange((status) => {
            setIsConnected(status);
        });

        return unsubscribe;
    }, [wsManager]);

    return (
        <header className="glass rounded-xl py-2 px-4 flex justify-between items-center bg-white/5 border-white/5 backdrop-blur-xl">
            {/* Logo */}
            <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center shadow-glow-sm">
                    <Brain className="w-5 h-5 text-white" />
                </div>
                <h1 className="text-lg md:text-xl font-bold tracking-tight">
                    Emotion<span className="gradient-text">AI</span>
                </h1>
            </div>

            {/* Connection Status */}
            <div className="flex items-center gap-2 px-2.5 py-1 bg-black/20 rounded-lg border border-white/5">
                <div className={`status-dot ${isConnected ? 'status-connected shadow-[0_0_8px_#10b981]' : 'status-disconnected'}`} />
                <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400">
                    {isConnected ? 'System Live' : 'Offline'}
                </span>
            </div>
        </header>
    );
}
