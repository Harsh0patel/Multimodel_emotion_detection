/**
 * Application Configuration
 * Matches test_frontend/app.py specifications
 */

export const CONFIG = {
    // WebSocket Configuration
    WS_URL: 'ws://127.0.0.1:8000/ws/stream',

    // Audio Configuration (matching test_frontend lines 30-34)
    AUDIO_CHUNK_SIZE: 4096,
    AUDIO_CHANNELS: 1,
    AUDIO_SAMPLE_RATE: 16000,
    BUFFER_DURATION: 5, // seconds

    // Video Configuration (matching test_frontend line 40)
    VIDEO_FPS: 5,

    // Send audio every 60 frames (~2 seconds at 30fps) (matching test_frontend line 157)
    SEND_AUDIO_INTERVAL: 5,

    // Face Detection Configuration (matching test_frontend line 73, 83)
    FACE_DETECTION_CONFIDENCE: 0.5,
    FACE_PADDING: 20,

    // Image Encoding (matching test_frontend line 147)
    JPEG_QUALITY: 0.85,

    // WebSocket Settings
    WS_PING_INTERVAL: null, // Disable ping/pong (matching test_frontend line 106)
    WS_CLOSE_TIMEOUT: 10000,
    WS_MAX_SIZE: 10_000_000, // 10MB (matching test_frontend line 108)
} as const;

export const EMOTION_ICONS: Record<string, string> = {
    happy: '😊',
    sad: '😢',
    angry: '😠',
    fear: '😨',
    surprise: '😲',
    disgust: '🤢',
    neutral: '😐',
    contempt: '😏',
    default: '🤔',
};

export const EMOTION_COLORS: Record<string, string> = {
    happy: 'from-yellow-400 to-orange-500',
    sad: 'from-blue-400 to-indigo-600',
    angry: 'from-red-500 to-pink-600',
    fear: 'from-purple-500 to-indigo-700',
    surprise: 'from-cyan-400 to-blue-500',
    disgust: 'from-green-500 to-teal-600',
    neutral: 'from-gray-400 to-gray-600',
    contempt: 'from-orange-500 to-red-600',
    default: 'from-purple-500 to-pink-500',
};
