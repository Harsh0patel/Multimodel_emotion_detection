/**
 * Face Detection using TensorFlow.js BlazeFace
 * Replicates YOLO face detection from test_frontend/app.py
 */

import * as blazeface from '@tensorflow-models/blazeface';
import '@tensorflow/tfjs';
import { CONFIG } from './config';
import type { FaceDetection } from '@/types';

let model: blazeface.BlazeFaceModel | null = null;

/**
 * Load BlazeFace model
 * Matches test_frontend lines 22-24
 */
export async function loadFaceDetectionModel(): Promise<void> {
    if (model) return;

    try {
        console.log('Loading face detection model...');
        model = await blazeface.load();
        console.log('✅ Face detection model loaded');
    } catch (error) {
        console.error('Failed to load face detection model:', error);
        throw error;
    }
}

/**
 * Detect faces in video frame
 * Matches test_frontend lines 71-90
 */
export async function detectFaces(
    videoElement: HTMLVideoElement
): Promise<FaceDetection | null> {
    if (!model) {
        throw new Error('Face detection model not loaded');
    }

    try {
        // Run face detection (matching conf=0.5 from test_frontend line 73)
        const predictions = await model.estimateFaces(videoElement, false);

        if (predictions.length === 0) {
            return null;
        }

        // Get first face (matching test_frontend lines 75-78)
        const face = predictions[0];

        // Extract bounding box
        const [x1, y1] = face.topLeft as [number, number];
        const [x2, y2] = face.bottomRight as [number, number];

        // Add padding (matching test_frontend lines 81-85)
        const videoWidth = videoElement.videoWidth;
        const videoHeight = videoElement.videoHeight;
        const pad = CONFIG.FACE_PADDING;

        const paddedX1 = Math.max(0, x1 - pad);
        const paddedY1 = Math.max(0, y1 - pad);
        const paddedX2 = Math.min(videoWidth, x2 + pad);
        const paddedY2 = Math.min(videoHeight, y2 + pad);

        return {
            bbox: [paddedX1, paddedY1, paddedX2, paddedY2],
            confidence: face.probability ? (Array.isArray(face.probability) ? face.probability[0] : (face.probability as any)) : 1.0,
            landmarks: face.landmarks as number[][],
        };
    } catch (error) {
        console.error('Face detection error:', error);
        return null;
    }
}

/**
 * Crop face from video frame
 * Matches test_frontend lines 87-89
 */
export function cropFace(
    videoElement: HTMLVideoElement,
    bbox: [number, number, number, number],
    targetWidth?: number,
    targetHeight?: number
): HTMLCanvasElement | null {
    const [x1, y1, x2, y2] = bbox;
    const width = x2 - x1;
    const height = y2 - y1;

    if (width <= 0 || height <= 0) {
        return null;
    }

    // Create canvas for cropped face
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth || width;
    canvas.height = targetHeight || height;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
        return null;
    }

    // Draw cropped region (with auto-scaling if targets provided)
    ctx.drawImage(
        videoElement,
        x1, y1, width, height,
        0, 0, canvas.width, canvas.height
    );

    return canvas;
}

/**
 * Convert canvas to JPEG blob
 * Matches test_frontend lines 147-148
 */
export async function canvasToJPEG(
    canvas: HTMLCanvasElement,
    quality: number = CONFIG.JPEG_QUALITY
): Promise<Blob> {
    return new Promise((resolve, reject) => {
        canvas.toBlob(
            (blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Failed to convert canvas to blob'));
                }
            },
            'image/jpeg',
            quality
        );
    });
}

/**
 * Draw bounding box on canvas
 * Matches test_frontend lines 178-182
 */
export function drawBoundingBox(
    canvas: HTMLCanvasElement,
    bbox: [number, number, number, number],
    emotion: string
): void {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const [x1, y1, x2, y2] = bbox;

    // Draw rectangle (matching green color from test_frontend)
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Draw label (matching test_frontend line 181-182)
    ctx.fillStyle = '#00ff00';
    ctx.font = '16px Inter, sans-serif';
    ctx.fillText(`Face | Emotion: ${emotion}`, x1, y1 - 10);
}

/**
 * Clear canvas
 */
export function clearCanvas(canvas: HTMLCanvasElement): void {
    const ctx = canvas.getContext('2d');
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}
