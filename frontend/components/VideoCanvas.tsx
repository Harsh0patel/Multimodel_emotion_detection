'use client';

import { useEffect, RefObject } from 'react';

interface VideoCanvasProps {
    videoRef: RefObject<HTMLVideoElement | null>;
    canvasRef: RefObject<HTMLCanvasElement | null>;
    isStreaming: boolean;
}

export default function VideoCanvas({ videoRef, canvasRef, isStreaming }: VideoCanvasProps) {
    // Sync canvas size with video
    useEffect(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (!video || !canvas) return;

        const handleLoadedMetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };

        video.addEventListener('loadedmetadata', handleLoadedMetadata);

        return () => {
            video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        };
    }, [videoRef, canvasRef]);

    return (
        <div className="video-container">
            {/* Video element */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="absolute inset-0 w-full h-full object-cover"
            />

            {/* Overlay canvas for bounding boxes */}
            <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full object-cover pointer-events-none z-10"
            />

            {/* Loading overlay */}
            {!isStreaming && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-md z-20 transition-all duration-700">
                    <div className="relative w-16 h-16 mb-4">
                        <div className="pulse-ring !border-neon-cyan/30" />
                        <div className="pulse-ring !border-white/10" />
                    </div>
                    <p className="text-[10px] text-white/50 font-black uppercase tracking-[0.3em] animate-pulse">
                        System Ready | Awaiting Input
                    </p>
                </div>
            )}
        </div>
    );
}
