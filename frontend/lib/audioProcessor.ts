/**
 * Audio Processor
 * Implements exact audio handling from test_frontend/app.py
 */

import { CONFIG } from './config';

/**
 * Audio Processor Class
 * Matches test_frontend lines 36-96
 */
export class AudioProcessor {
    private audioContext: AudioContext | null = null;
    private mediaStreamSource: MediaStreamAudioSourceNode | null = null;
    private scriptProcessor: ScriptProcessorNode | null = null;
    private analyser: AnalyserNode | null = null;

    // Audio buffer matching test_frontend line 43
    private audioBuffer: Int16Array[] = [];
    private maxBufferLength: number;

    constructor() {
        // Calculate max buffer length (matching test_frontend line 43)
        // maxlen=int(RATE * BUFFER_DURATION / CHUNK) + 1
        this.maxBufferLength = Math.floor(
            (CONFIG.AUDIO_SAMPLE_RATE * CONFIG.BUFFER_DURATION) / CONFIG.AUDIO_CHUNK_SIZE
        ) + 1;
    }

    /**
     * Initialize audio capture
     * Matches test_frontend lines 45-54
     */
    async init(stream: MediaStream): Promise<void> {
        try {
            // Create audio context with specific sample rate
            this.audioContext = new AudioContext({
                sampleRate: CONFIG.AUDIO_SAMPLE_RATE,
            });

            // Create media stream source
            this.mediaStreamSource = this.audioContext.createMediaStreamSource(stream);

            // Create analyser for visualization
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;

            // Create script processor for audio data capture
            this.scriptProcessor = this.audioContext.createScriptProcessor(
                CONFIG.AUDIO_CHUNK_SIZE,
                CONFIG.AUDIO_CHANNELS,
                CONFIG.AUDIO_CHANNELS
            );

            // Audio callback (matching test_frontend lines 56-58)
            this.scriptProcessor.onaudioprocess = (event) => {
                const inputData = event.inputBuffer.getChannelData(0);

                // Convert Float32 to Int16 (PCM format)
                const int16Data = this.float32ToInt16(inputData);

                // Add to buffer (matching deque behavior)
                this.audioBuffer.push(int16Data);

                // Maintain max buffer length (deque maxlen behavior)
                if (this.audioBuffer.length > this.maxBufferLength) {
                    this.audioBuffer.shift();
                }
            };

            // Connect audio nodes
            this.mediaStreamSource.connect(this.analyser);
            this.mediaStreamSource.connect(this.scriptProcessor);
            this.scriptProcessor.connect(this.audioContext.destination);

            console.log('✅ Audio processor initialized');
        } catch (error) {
            console.error('Failed to initialize audio processor:', error);
            throw error;
        }
    }

    /**
     * Convert Float32Array to Int16Array (PCM format)
     * Matches pyaudio.paInt16 format from test_frontend
     */
    private float32ToInt16(float32Array: Float32Array): Int16Array {
        const int16Array = new Int16Array(float32Array.length);

        for (let i = 0; i < float32Array.length; i++) {
            // Clamp value between -1 and 1
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            // Convert to 16-bit PCM
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        return int16Array;
    }

    /**
     * Get buffered audio chunk
     * Matches test_frontend lines 60-64
     */
    getAudioChunk(): ArrayBuffer | null {
        if (this.audioBuffer.length === 0) {
            return null;
        }

        // Join all buffered chunks (matching b''.join(list(self.audio_buffer)))
        const totalLength = this.audioBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
        const combined = new Int16Array(totalLength);

        let offset = 0;
        for (const chunk of this.audioBuffer) {
            combined.set(chunk, offset);
            offset += chunk.length;
        }

        return combined.buffer;
    }

    /**
     * Get analyser for visualization
     */
    getAnalyser(): AnalyserNode | null {
        return this.analyser;
    }

    /**
     * Get current buffer status
     */
    getBufferStatus(): { current: number; max: number } {
        return {
            current: this.audioBuffer.length,
            max: this.maxBufferLength,
        };
    }

    /**
     * Clear audio buffer
     */
    clearBuffer(): void {
        this.audioBuffer = [];
    }

    /**
     * Stop and cleanup
     * Matches test_frontend lines 92-96
     */
    stop(): void {
        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
            this.scriptProcessor = null;
        }

        if (this.mediaStreamSource) {
            this.mediaStreamSource.disconnect();
            this.mediaStreamSource = null;
        }

        if (this.analyser) {
            this.analyser.disconnect();
            this.analyser = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        this.audioBuffer = [];
        console.log('🧹 Audio processor stopped');
    }
}
