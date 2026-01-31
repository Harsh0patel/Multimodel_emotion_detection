'use client';

import { useEffect, useRef } from 'react';
import type { AudioProcessor } from '@/lib/audioProcessor';

interface AudioVisualizerProps {
    audioProcessor: AudioProcessor | null;
}

export default function AudioVisualizer({ audioProcessor }: AudioVisualizerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationRef = useRef<number | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !audioProcessor) {
            return;
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const analyser = audioProcessor.getAnalyser();
        if (!analyser) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            animationRef.current = requestAnimationFrame(draw);

            analyser.getByteFrequencyData(dataArray);

            const width = canvas.width;
            const height = canvas.height;

            ctx.clearRect(0, 0, width, height);

            const barWidth = (width / bufferLength) * 2.5;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * height;

                // Create gradient
                const gradient = ctx.createLinearGradient(0, height - barHeight, 0, height);
                gradient.addColorStop(0, '#667eea');
                gradient.addColorStop(1, '#764ba2');

                ctx.fillStyle = gradient;
                ctx.fillRect(x, height - barHeight, barWidth, barHeight);

                x += barWidth + 1;
            }
        };

        draw();

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [audioProcessor]);

    return (
        <div className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
                Audio Input
            </h3>
            <div className="bg-black/20 rounded-xl p-4">
                <canvas
                    ref={canvasRef}
                    width={350}
                    height={80}
                    className="w-full h-20 rounded"
                />
            </div>
        </div>
    );
}
