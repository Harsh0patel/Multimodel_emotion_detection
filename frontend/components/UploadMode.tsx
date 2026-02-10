'use client';

import { useState, useRef, useEffect } from 'react';
import { Upload, Play, FileVideo, Square } from 'lucide-react';
import { getWebSocketManager } from '@/lib/websocket';
import { loadFaceDetectionModel, detectFaces, cropFace, canvasToJPEG, drawBoundingBox, clearCanvas } from '@/lib/faceDetection';
import { CONFIG } from '@/lib/config';
import type { LogEntry } from '@/types';
import ResultsPanel from './ResultsPanel';

export default function UploadMode() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [emotion, setEmotion] = useState('Upload a video to begin');
    const [stats, setStats] = useState({ frames: 0, audioChunks: 0 });

    const fileInputRef = useRef<HTMLInputElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const wsManagerRef = useRef(getWebSocketManager());
    const [isDragging, setIsDragging] = useState(false);
    const isActiveRef = useRef(false);
    const currentEmotionRef = useRef('Detecting...');

    const addLog = (message: string, level: LogEntry['level'] = 'info') => {
        const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
        setLogs(prev => [{ timestamp, message, level }, ...prev].slice(0, 50));
    };

    const handleFileSelect = (file: File) => {
        if (!file.type.startsWith('video/')) {
            addLog('Please select a valid video file', 'error');
            return;
        }

        setSelectedFile(file);
        const url = URL.createObjectURL(file);
        setPreviewUrl(url);
        addLog(`File loaded: ${file.name}`, 'success');
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const file = e.dataTransfer.files[0];
        if (file) {
            handleFileSelect(file);
        }
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const extractAudioData = async (file: File): Promise<Int16Array> => {
        const AudioContextClass = (window.AudioContext || (window as any).webkitAudioContext);
        const audioContext = new AudioContextClass();
        const arrayBuffer = await file.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const offlineContext = new OfflineAudioContext(
            1,
            audioBuffer.duration * CONFIG.AUDIO_SAMPLE_RATE,
            CONFIG.AUDIO_SAMPLE_RATE
        );

        const source = offlineContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(offlineContext.destination);
        source.start();

        const renderedBuffer = await offlineContext.startRendering();
        const float32Data = renderedBuffer.getChannelData(0);

        const int16Data = new Int16Array(float32Data.length);
        for (let i = 0; i < float32Data.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Data[i]));
            int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        return int16Data;
    };

    const processVideo = async () => {
        if (!selectedFile || !videoRef.current) return;

        try {
            isActiveRef.current = true;
            setIsProcessing(true);
            setProgress(0);
            setStats({ frames: 0, audioChunks: 0 });
            addLog('Starting video processing...', 'info');

            // 1. Load model
            addLog('Loading face detection model...', 'info');
            await loadFaceDetectionModel();

            // 2. Extract Audio
            addLog('Extracting audio track...', 'info');
            const pcmData = await extractAudioData(selectedFile);
            const totalAudioLength = pcmData.length;

            // 3. Connect WebSocket
            addLog('Connecting to backend...', 'info');
            await wsManagerRef.current.connect();

            const unsubscribe = wsManagerRef.current.onMessage((msg) => {
                if (msg.type === 'result') {
                    const emotionStr = typeof msg.emotion === 'object'
                        ? (msg.emotion as any).emotion
                        : msg.emotion;
                    setEmotion(emotionStr);
                    currentEmotionRef.current = emotionStr;
                    addLog(`Emotion detected: ${emotionStr}`, 'success');
                }
            });

            // 4. Process Video Frames
            const video = videoRef.current;

            if (isNaN(video.duration) || video.duration === 0) {
                addLog('Waiting for video metadata...', 'info');
                await new Promise(resolve => {
                    if (video.readyState >= 1) resolve(null);
                    video.onloadedmetadata = () => resolve(null);
                });
            }

            const duration = video.duration;
            const targetFPS = 10; // Process at 10 FPS for significantly faster upload results
            const interval = 1 / targetFPS;

            let currentFrame = 0;
            let currentAudioOffset = 0;
            const audioInterval = 5; // Send audio every 5 steps (0.5s at 10 FPS)
            const audioStepSize = Math.floor(CONFIG.AUDIO_SAMPLE_RATE * 0.5);

            addLog(`Processing at ${targetFPS} FPS (${Math.floor(duration * targetFPS)} frames total)...`, 'info');

            for (let time = 0; time < duration && isActiveRef.current; time += interval) {
                video.currentTime = time;
                await new Promise(resolve => {
                    video.onseeked = resolve;
                });

                // Clear previous drawings
                if (canvasRef.current) {
                    clearCanvas(canvasRef.current);
                }

                const faceDetection = await detectFaces(video);
                if (faceDetection) {
                    // Draw bounding box
                    if (canvasRef.current) {
                        drawBoundingBox(canvasRef.current, faceDetection.bbox, currentEmotionRef.current);
                    }

                    const faceCanvas = cropFace(video, faceDetection.bbox, 224, 224);
                    if (faceCanvas && wsManagerRef.current.isConnected()) {
                        const blob = await canvasToJPEG(faceCanvas);
                        const buffer = await blob.arrayBuffer();
                        try {
                            await wsManagerRef.current.sendFrame(buffer);
                        } catch (err) {
                            console.error('Failed to send frame:', err);
                        }
                    }
                }

                currentFrame++;

                if (currentFrame > 0 && currentFrame % audioInterval === 0) {
                    const endOffset = Math.min(currentAudioOffset + audioStepSize, totalAudioLength);
                    if (currentAudioOffset < totalAudioLength && wsManagerRef.current.isConnected()) {
                        const chunk = pcmData.slice(currentAudioOffset, endOffset);
                        try {
                            await wsManagerRef.current.sendAudio(chunk.buffer);
                            currentAudioOffset = endOffset;
                            setStats(prev => ({ ...prev, audioChunks: prev.audioChunks + 1 }));
                        } catch (err) {
                            console.error('Failed to send audio:', err);
                        }
                        await new Promise(resolve => setTimeout(resolve, 200));
                    }
                }

                const currentProgress = Math.min(100, Math.round((time / duration) * 100));
                setProgress(currentProgress);
                setStats(prev => ({ ...prev, frames: currentFrame }));
            }

            unsubscribe();
            if (canvasRef.current) clearCanvas(canvasRef.current);
            addLog('Processing complete', 'success');
            setIsProcessing(false);
            setProgress(100);
            isActiveRef.current = false;
            await wsManagerRef.current.close();

        } catch (error: any) {
            const errMsg = error instanceof Error ? error.message : 'An unknown error occurred during processing.';
            addLog(`Processing failed: ${errMsg}`, 'error');
            setIsProcessing(false);
            isActiveRef.current = false;
            await wsManagerRef.current.close();
        }
    };

    const stopProcessing = () => {
        isActiveRef.current = false;
        setIsProcessing(false);
        if (canvasRef.current) clearCanvas(canvasRef.current);
        addLog('Processing stopped by user', 'warning');
    };

    useEffect(() => {
        return () => {
            isActiveRef.current = false;
            wsManagerRef.current.close();
        };
    }, []);

    return (
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-4">
            <div className="space-y-3">
                {!selectedFile ? (
                    <div
                        onClick={() => fileInputRef.current?.click()}
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        className={`video-container flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${isDragging ? 'border-neon-purple bg-neon-purple/10 scale-[1.02]' : 'hover:border-cyber-border-glow'}`}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="video/*"
                            onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                            className="hidden"
                        />
                        <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mb-4 shadow-glow-sm transform group-hover:scale-110 transition-transform duration-500">
                            <Upload className="w-8 h-8 text-white" />
                        </div>
                        <h3 className="text-xl font-black uppercase tracking-tighter mb-1">Select Video</h3>
                        <p className="text-[10px] text-gray-500 font-bold uppercase tracking-widest mb-4">DRAG & DROP OR BROWSE</p>
                        <button className="px-6 py-2 rounded-full bg-white/5 border border-white/10 text-[10px] font-black uppercase tracking-widest hover:bg-white/10 transition-all">Choose File</button>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <div className="video-container">
                            <video
                                ref={videoRef}
                                src={previewUrl || undefined}
                                controls={!isProcessing}
                                className="w-full h-full object-contain"
                                onLoadedMetadata={(e) => {
                                    if (canvasRef.current) {
                                        canvasRef.current.width = e.currentTarget.videoWidth;
                                        canvasRef.current.height = e.currentTarget.videoHeight;
                                    }
                                }}
                            />
                            <canvas
                                ref={canvasRef}
                                className="absolute top-0 left-0 w-full h-full object-contain pointer-events-none z-10"
                            />
                            {isProcessing && (
                                <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full flex items-center gap-2 z-20 border border-white/10">
                                    <div className="w-2 h-2 bg-neon-cyan rounded-full animate-pulse" />
                                    <span className="text-[10px] font-bold text-neon-cyan tracking-widest uppercase">Processing {progress}%</span>
                                </div>
                            )}
                        </div>
                        <div className="glass rounded-xl px-4 py-2 flex items-center justify-between border-white/5 bg-white/5">
                            <div className="flex items-center gap-2">
                                <FileVideo className="w-4 h-4 text-neon-cyan" />
                                <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400 truncate max-w-[150px]">{selectedFile.name}</span>
                            </div>
                            <div className="flex gap-2">
                                {isProcessing ? (
                                    <button onClick={stopProcessing} className="btn-secondary py-1.5 px-4 rounded-full text-[10px] uppercase font-black tracking-widest flex items-center gap-2">
                                        <Square size={12} />
                                        <span>Stop</span>
                                    </button>
                                ) : (
                                    <>
                                        <button onClick={() => setSelectedFile(null)} className="text-[10px] font-black uppercase tracking-widest text-gray-400 hover:text-white transition-colors">Change</button>
                                        <button onClick={processVideo} className="btn-success py-1.5 px-4 rounded-full text-[10px] uppercase font-black tracking-widest flex items-center gap-2 shadow-glow-sm">
                                            <Play className="w-3 h-3" />
                                            <span>Process Video</span>
                                        </button>
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            <ResultsPanel
                emotion={emotion}
                framesProcessed={stats.frames}
                audioChunks={stats.audioChunks}
                sessionStartTime={null}
                logs={logs}
                audioProcessor={null}
            />
        </div>
    );
}
