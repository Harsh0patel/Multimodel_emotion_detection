/**
 * Media Capture Utilities
 * Handles video and audio stream capture
 */

import { CONFIG } from './config';

/**
 * Get user media stream
 * Matches test_frontend lines 38-40, 45-53
 */
export async function getUserMedia(): Promise<MediaStream> {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user',
                frameRate: { ideal: CONFIG.VIDEO_FPS },
            },
            audio: {
                channelCount: CONFIG.AUDIO_CHANNELS,
                sampleRate: CONFIG.AUDIO_SAMPLE_RATE,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        console.log('✅ Media stream acquired');
        return stream;
    } catch (error) {
        console.error('Failed to get user media:', error);
        throw error;
    }
}

/**
 * Stop all tracks in a media stream
 */
export function stopMediaStream(stream: MediaStream): void {
    stream.getTracks().forEach(track => {
        track.stop();
        console.log(`Stopped ${track.kind} track`);
    });
}

/**
 * Check if browser supports required APIs
 */
export function checkBrowserSupport(): {
    supported: boolean;
    missing: string[];
} {
    const missing: string[] = [];

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        missing.push('getUserMedia');
    }

    if (!window.WebSocket) {
        missing.push('WebSocket');
    }

    if (!window.AudioContext && !(window as any).webkitAudioContext) {
        missing.push('AudioContext');
    }

    return {
        supported: missing.length === 0,
        missing,
    };
}

/**
 * Format session time
 */
export function formatSessionTime(milliseconds: number): string {
    const totalSeconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

/**
 * Get current timestamp for logging
 */
export function getTimestamp(): string {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { hour12: false });
}

/**
 * Concatenate array buffers
 */
export function concatenateBuffers(buffers: ArrayBuffer[]): ArrayBuffer {
    const totalLength = buffers.reduce((sum, buf) => sum + buf.byteLength, 0);
    const result = new Uint8Array(totalLength);

    let offset = 0;
    for (const buffer of buffers) {
        result.set(new Uint8Array(buffer), offset);
        offset += buffer.byteLength;
    }

    return result.buffer;
}
