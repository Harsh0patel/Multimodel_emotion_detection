/**
 * TypeScript Type Definitions
 */

export interface EmotionResult {
    type: 'result';
    emotion: string;
    frames_processed?: number;
    audio_chunks?: number;
    timestamp: number;
}

export interface ErrorMessage {
    type: 'error';
    message: string;
    timestamp: number;
}

export type WebSocketMessage = EmotionResult | ErrorMessage;

export interface FaceDetection {
    bbox: [number, number, number, number]; // [x1, y1, x2, y2]
    confidence: number;
    landmarks?: number[][];
}

export interface MediaState {
    isStreaming: boolean;
    isConnected: boolean;
    currentEmotion: string;
    frameCount: number;
    audioChunkCount: number;
    sessionStartTime: number | null;
    error: string | null;
}

export interface AudioBufferState {
    buffer: Int16Array[];
    maxLength: number;
    currentLength: number;
}

export interface StreamStats {
    framesProcessed: number;
    audioChunks: number;
    sessionTime: string;
    fps: number;
}

export type LogLevel = 'info' | 'success' | 'warning' | 'error';

export interface LogEntry {
    timestamp: string;
    message: string;
    level: LogLevel;
}

export type AppMode = 'stream' | 'upload';
