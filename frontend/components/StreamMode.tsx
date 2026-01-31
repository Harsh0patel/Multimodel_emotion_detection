'use client';

import { useState, useEffect, useRef } from 'react';
import { Play, Square } from 'lucide-react';
import { getWebSocketManager } from '@/lib/websocket';
import { AudioProcessor } from '@/lib/audioProcessor';
import { getUserMedia, stopMediaStream, formatSessionTime } from '@/lib/mediaCapture';
import { loadFaceDetectionModel, detectFaces, cropFace, canvasToJPEG, drawBoundingBox, clearCanvas } from '@/lib/faceDetection';
import { CONFIG } from '@/lib/config';
import type { MediaState, LogEntry } from '@/types';
import VideoCanvas from './VideoCanvas';
import ResultsPanel from './ResultsPanel';

export default function StreamMode() {
    const [mediaState, setMediaState] = useState<MediaState>({
        isStreaming: false,
        isConnected: false,
        currentEmotion: 'Waiting...',
        frameCount: 0,
        audioChunkCount: 0,
        sessionStartTime: null,
        error: null,
    });

    const [logs, setLogs] = useState<LogEntry[]>([]);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const audioProcessorRef = useRef<AudioProcessor | null>(null);
    const wsManagerRef = useRef(getWebSocketManager());
    const animationFrameRef = useRef<number | null>(null);
    const frameCountRef = useRef(0);
    const isStreamingRef = useRef(false);
    const lastAudioSendTimeRef = useRef<number>(0);
    const currentEmotionRef = useRef('Waiting...');

    // Add log entry
    const addLog = (message: string, level: LogEntry['level'] = 'info') => {
        const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
        setLogs(prev => [{ timestamp, message, level }, ...prev].slice(0, 50));
    };

    // Load face detection model on mount
    useEffect(() => {
        loadFaceDetectionModel().catch(err => {
            addLog(`Failed to load face detection model: ${err.message}`, 'error');
        });
    }, []);

    // Start streaming
    const startStreaming = async () => {
        try {
            addLog('Initializing...', 'info');

            // Ensure face detection model is loaded
            try {
                await loadFaceDetectionModel();
            } catch (err: any) {
                throw new Error(`Face detection model failed: ${err.message}`);
            }

            // 1. Get media stream (Do this BEFORE connecting to avoid idle time gaps)
            addLog('Requesting camera and microphone access...', 'info');
            const stream = await getUserMedia();
            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }

            addLog('Camera and microphone access granted', 'success');

            // 2. Connect to WebSocket
            addLog('Connecting to server...', 'info');
            await wsManagerRef.current.connect();

            setMediaState(prev => ({ ...prev, isConnected: true }));
            addLog('Connected to server', 'success');

            // Initialize audio processor
            const audioProcessor = new AudioProcessor();
            await audioProcessor.init(stream);
            audioProcessorRef.current = audioProcessor;

            wsManagerRef.current.onMessage((message) => {
                if (message.type === 'result') {
                    const emotionStr = typeof message.emotion === 'object'
                        ? (message.emotion as any).emotion
                        : message.emotion;

                    setMediaState(prev => ({
                        ...prev,
                        currentEmotion: emotionStr,
                    }));
                    currentEmotionRef.current = emotionStr;
                    addLog(`Detected emotion: ${emotionStr}`, 'success');
                } else if (message.type === 'error') {
                    addLog(`Error: ${message.message}`, 'error');
                }
            });

            // Start streaming
            isStreamingRef.current = true;
            setMediaState(prev => ({
                ...prev,
                isStreaming: true,
                sessionStartTime: Date.now(),
                frameCount: 0,
                audioChunkCount: 0,
            }));

            frameCountRef.current = 0;
            lastAudioSendTimeRef.current = Date.now();
            addLog('Streaming started', 'success');

            // Start capture loop
            captureLoop();

        } catch (error: any) {
            addLog(`Failed to start: ${error.message}`, 'error');
            stopStreaming();
        }
    };

    // Capture loop (matches test_frontend lines 130-189)
    const captureLoop = async () => {
        if (!isStreamingRef.current || !videoRef.current || !canvasRef.current) {
            return;
        }

        try {
            const video = videoRef.current;
            const canvas = canvasRef.current;

            // Increment frame count every loop to keep timing consistent for audio
            frameCountRef.current++;
            setMediaState(prev => ({ ...prev, frameCount: frameCountRef.current }));

            let faceDetection = null;
            try {
                // Detect face
                faceDetection = await detectFaces(video);
            } catch (err) {
                console.warn('Face detection skipped: Model not ready');
            }

            if (faceDetection) {
                // Crop face (matching test_frontend lines 87-89)
                const faceCanvas = cropFace(video, faceDetection.bbox, 224, 224);

                if (faceCanvas && wsManagerRef.current.isConnected()) {
                    // Convert to JPEG (matching test_frontend lines 147-148)
                    const blob = await canvasToJPEG(faceCanvas);
                    const arrayBuffer = await blob.arrayBuffer();

                    // Send frame (matching test_frontend line 151)
                    try {
                        await wsManagerRef.current.sendFrame(arrayBuffer);
                    } catch (err) {
                        console.error('Failed to send frame:', err);
                    }

                    // Draw bounding box on overlay canvas
                    clearCanvas(canvas);
                    drawBoundingBox(canvas, faceDetection.bbox, currentEmotionRef.current);
                }
            } else {
                // Clear canvas if no face detected
                clearCanvas(canvas);
            }

            // Send audio every 2 seconds of real time
            // Replaces frame-based timing (frameCountRef.current % 60)
            const now = Date.now();
            if (now - lastAudioSendTimeRef.current >= 2000) {
                const audioChunk = audioProcessorRef.current?.getAudioChunk();

                if (audioChunk && wsManagerRef.current.isConnected()) {
                    try {
                        await wsManagerRef.current.sendAudio(audioChunk);
                        setMediaState(prev => ({ ...prev, audioChunkCount: prev.audioChunkCount + 1 }));
                        addLog('Sent audio chunk', 'info');
                        lastAudioSendTimeRef.current = now;
                    } catch (err) {
                        console.error('Failed to send audio:', err);
                    }
                }
            }

        } catch (error: any) {
            console.error('Capture error:', error);
        }

        // Schedule next frame
        if (isStreamingRef.current) {
            animationFrameRef.current = requestAnimationFrame(() => {
                setTimeout(captureLoop, 1000 / CONFIG.VIDEO_FPS);
            });
        }
    };

    // Stop streaming
    const stopStreaming = () => {
        // Cancel animation frame
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }

        // Stop audio processor
        if (audioProcessorRef.current) {
            audioProcessorRef.current.stop();
            audioProcessorRef.current = null;
        }

        // Stop media stream
        if (streamRef.current) {
            stopMediaStream(streamRef.current);
            streamRef.current = null;
        }

        // Close WebSocket
        wsManagerRef.current.close();

        isStreamingRef.current = false;

        // Clear canvas
        if (canvasRef.current) {
            clearCanvas(canvasRef.current);
        }

        setMediaState({
            isStreaming: false,
            isConnected: false,
            currentEmotion: 'Waiting...',
            frameCount: 0,
            audioChunkCount: 0,
            sessionStartTime: null,
            error: null,
        });

        frameCountRef.current = 0;
        currentEmotionRef.current = 'Waiting...';
        addLog('Streaming stopped', 'warning');
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            stopStreaming();
        };
    }, []);

    return (
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-4">
            {/* Video Section */}
            <div className="space-y-3">
                <VideoCanvas
                    videoRef={videoRef}
                    canvasRef={canvasRef}
                    isStreaming={mediaState.isStreaming}
                />

                {/* Controls */}
                <div className="flex gap-2">
                    <button
                        onClick={startStreaming}
                        disabled={mediaState.isStreaming}
                        className="btn-success flex-1 flex items-center justify-center gap-2 py-2.5 text-xs uppercase tracking-widest font-black"
                    >
                        <Play className="w-4 h-4" />
                        <span>Start Detection</span>
                    </button>

                    <button
                        onClick={stopStreaming}
                        disabled={!mediaState.isStreaming}
                        className="btn-secondary flex-1 flex items-center justify-center gap-2 py-2.5 text-xs uppercase tracking-widest font-black"
                    >
                        <Square className="w-4 h-4" />
                        <span>Stop Session</span>
                    </button>
                </div>
            </div>

            {/* Results Panel */}
            <ResultsPanel
                emotion={mediaState.currentEmotion}
                framesProcessed={mediaState.frameCount}
                audioChunks={mediaState.audioChunkCount}
                sessionStartTime={mediaState.sessionStartTime}
                logs={logs}
                audioProcessor={audioProcessorRef.current}
            />
        </div>
    );
}
