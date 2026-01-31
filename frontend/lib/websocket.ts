/**
 * WebSocket Manager
 * Implements exact protocol from test_frontend/app.py
 */

import { CONFIG } from './config';
import type { WebSocketMessage } from '@/types';

export class WebSocketManager {
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 2000;
    private messageHandlers: Set<(message: WebSocketMessage) => void> = new Set();
    private statusHandlers: Set<(connected: boolean) => void> = new Set();

    constructor(private url: string = CONFIG.WS_URL) { }

    /**
     * Connect to WebSocket server
     * Matches test_frontend lines 104-109
     */
    async connect(): Promise<void> {
        // If already connected, resolve immediately
        if (this.isConnected()) {
            return Promise.resolve();
        }

        // If connecting, wait for it or close and restart
        if (this.ws && (this.ws.readyState === WebSocket.CONNECTING)) {
            return new Promise((resolve, reject) => {
                const check = () => {
                    if (this.isConnected()) resolve();
                    else if (this.ws?.readyState === WebSocket.CLOSED) this.connect().then(resolve).catch(reject);
                    else setTimeout(check, 100);
                };
                check();
            });
        }

        return new Promise((resolve, reject) => {
            try {
                // Ensure old connection is cleaned up
                if (this.ws) {
                    this.ws.onopen = null;
                    this.ws.onerror = null;
                    this.ws.onclose = null;
                    this.ws.onmessage = null;
                    this.ws.close();
                }

                this.ws = new WebSocket(this.url);
                this.ws.binaryType = 'arraybuffer';

                this.ws.onopen = () => {
                    console.log('✅ WebSocket connected');
                    this.reconnectAttempts = 0;
                    this.notifyStatus(true);
                    resolve();
                };

                this.ws.onerror = (error) => {
                    console.error('❌ WebSocket error:', error);
                    reject(error);
                };

                this.ws.onclose = () => {
                    console.log('⚪ WebSocket disconnected');
                    this.notifyStatus(false);
                    this.handleReconnect();
                };

                this.ws.onmessage = (event) => {
                    this.handleMessage(event.data);
                };
            } catch (error) {
                reject(error);
            }
        });
    }

    /**
     * Handle incoming messages
     * Matches test_frontend lines 121-124
     */
    private handleMessage(data: string) {
        try {
            const message: WebSocketMessage = JSON.parse(data);
            this.messageHandlers.forEach(handler => handler(message));
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }

    /**
     * Send video frame with FRAME header
     * Matches test_frontend lines 147-151
     */
    async sendFrame(frameData: ArrayBuffer): Promise<void> {
        if (!this.isConnected()) {
            throw new Error('WebSocket not connected');
        }

        try {
            // Create message with FRAME header (5 bytes)
            const header = new TextEncoder().encode('FRAME');
            const message = new Uint8Array(header.length + frameData.byteLength);
            message.set(header, 0);
            message.set(new Uint8Array(frameData), header.length);

            this.ws!.send(message.buffer);
            // console.debug('📤 Sent FRAME');
        } catch (error) {
            console.error('Failed to send frame:', error);
            throw error;
        }
    }

    /**
     * Send audio chunk with AUDIO header
     * Matches test_frontend lines 162-163
     */
    async sendAudio(audioData: ArrayBuffer): Promise<void> {
        if (!this.isConnected()) {
            throw new Error('WebSocket not connected');
        }

        try {
            // Create message with AUDIO header (5 bytes)
            const header = new TextEncoder().encode('AUDIO');
            const message = new Uint8Array(header.length + audioData.byteLength);
            message.set(header, 0);
            message.set(new Uint8Array(audioData), header.length);

            this.ws!.send(message.buffer);
            console.log('📤 Sent AUDIO chunk');
        } catch (error) {
            console.error('Failed to send audio:', error);
            throw error;
        }
    }

    /**
     * Subscribe to WebSocket messages
     */
    onMessage(handler: (message: WebSocketMessage) => void): () => void {
        this.messageHandlers.add(handler);
        return () => this.messageHandlers.delete(handler);
    }

    /**
     * Subscribe to connection status changes
     */
    onStatusChange(handler: (connected: boolean) => void): () => void {
        this.statusHandlers.add(handler);
        return () => this.statusHandlers.delete(handler);
    }

    /**
     * Notify all status handlers
     */
    private notifyStatus(connected: boolean): void {
        this.statusHandlers.forEach(handler => handler(connected));
    }

    /**
     * Handle reconnection logic
     */
    private handleReconnect(): void {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);

            setTimeout(() => {
                this.connect().catch(console.error);
            }, this.reconnectDelay * this.reconnectAttempts);
        }
    }

    /**
     * Check if WebSocket is connected
     */
    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }

    /**
     * Close WebSocket connection
     * Matches test_frontend lines 203-206
     */
    async close(): Promise<void> {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
            this.notifyStatus(false);
        }
    }

    /**
     * Reset reconnection attempts
     */
    resetReconnection(): void {
        this.reconnectAttempts = 0;
    }
}

// Singleton instance
let wsInstance: WebSocketManager | null = null;

export function getWebSocketManager(): WebSocketManager {
    if (!wsInstance) {
        wsInstance = new WebSocketManager();
    }
    return wsInstance;
}
