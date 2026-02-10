# EmotionAI - Next.js Frontend

A futuristic, modern Next.js frontend for real-time multimodal emotion detection using video and audio analysis.

## ✨ Features

### 🎥 Live Stream Mode
- **Real-time video streaming** from webcam with face detection
- **Audio capture** at 16kHz PCM Int16 format
- **Face detection** using TensorFlow.js BlazeFace model
- **WebSocket communication** with exact protocol matching test_frontend
- **Live audio visualizer** with frequency spectrum display
- **Real-time statistics** (frames, audio chunks, session time)
- **Activity log** with color-coded entries

### 📤 Upload Video Mode
- **Drag-and-drop** file upload
- **Video preview** before processing
- **Progress tracking** during analysis

### 🎨 Futuristic UI/UX
- **Cyberpunk-inspired** dark theme
- **Glassmorphism** effects with backdrop blur
- **Neon gradients** and glow effects
- **Smooth animations** and transitions
- **Responsive design** for all screen sizes
- **Custom fonts** (Inter + JetBrains Mono)

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Backend server running at `http://localhost:8000`
- Modern web browser with camera/microphone support

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

```bash
npm run build
npm start
```

## 📋 How It Works

### Backend Protocol (Matching test_frontend/app.py)

The frontend implements the exact same protocol as the test_frontend:

#### Video Frame Sending
```typescript
// Send face crop as JPEG with "FRAME" header
const header = new TextEncoder().encode('FRAME');
const message = new Uint8Array(header.length + jpegData.byteLength);
message.set(header, 0);
message.set(new Uint8Array(jpegData), header.length);
websocket.send(message.buffer);
```

#### Audio Sending
```typescript
// Send audio every 60 frames (~2 seconds at 30fps)
if (frameCount % 60 === 0) {
  const header = new TextEncoder().encode('AUDIO');
  const message = new Uint8Array(header.length + audioData.byteLength);
  message.set(header, 0);
  message.set(new Uint8Array(audioData), header.length);
  websocket.send(message.buffer);
}
```

#### Receiving Results
```typescript
// JSON response from backend
{
  "type": "result",
  "emotion": "happy",
  "frames_processed": 60,
  "audio_chunks": 1,
  "timestamp": 1234567890.123
}
```

### Technical Implementation

#### Face Detection
- Uses TensorFlow.js BlazeFace model (YOLO equivalent)
- Detects faces with confidence > 0.5
- Adds 20px padding to bounding box
- Crops and sends only face region

#### Audio Processing
- Captures at 16kHz, mono, PCM Int16
- 2-second circular buffer (deque pattern)
- Converts Float32 to Int16 for backend compatibility
- Sends buffered audio every 2 seconds

#### WebSocket Manager
- Automatic reconnection with exponential backoff
- Binary message support
- Message queue for reliability
- Connection status tracking

## 🎯 Configuration

Edit `lib/config.ts` to customize:

```typescript
export const CONFIG = {
  WS_URL: 'ws://localhost:8000/ws/stream',
  AUDIO_CHUNK_SIZE: 4096,
  AUDIO_CHANNELS: 1,
  AUDIO_SAMPLE_RATE: 16000,
  BUFFER_DURATION: 2,
  VIDEO_FPS: 30,
  SEND_AUDIO_INTERVAL: 60,
  FACE_DETECTION_CONFIDENCE: 0.5,
  FACE_PADDING: 20,
  JPEG_QUALITY: 0.85,
};
```

## 📁 Project Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout with fonts and background
│   ├── page.tsx            # Main page with mode switching
│   └── globals.css         # Global styles and utilities
├── components/
│   ├── Header.tsx          # Header with logo and status
│   ├── StreamMode.tsx      # Live streaming interface
│   ├── UploadMode.tsx      # Video upload interface
│   ├── VideoCanvas.tsx     # Video display with overlay
│   ├── ResultsPanel.tsx    # Results and statistics
│   └── AudioVisualizer.tsx # Audio frequency visualizer
├── lib/
│   ├── config.ts           # Application configuration
│   ├── websocket.ts        # WebSocket manager
│   ├── audioProcessor.ts   # Audio capture and processing
│   ├── faceDetection.ts    # Face detection with TensorFlow.js
│   └── mediaCapture.ts     # Media stream utilities
├── types/
│   └── index.ts            # TypeScript type definitions
└── tailwind.config.ts      # Tailwind theme configuration
```

## 🎨 Customization

### Change Colors

Edit `tailwind.config.ts`:

```typescript
colors: {
  neon: {
    cyan: '#00f2fe',
    purple: '#667eea',
    // Add your colors...
  },
}
```

### Add Emotions

Edit `lib/config.ts`:

```typescript
export const EMOTION_ICONS: Record<string, string> = {
  happy: '😊',
  sad: '😢',
  // Add more...
};
```

## 🐛 Troubleshooting

### WebSocket Connection Failed
- Ensure backend is running at `http://localhost:8000`
- Check browser console for errors
- Verify CORS settings in backend

### Camera/Microphone Access Denied
- Grant permissions in browser settings
- Use HTTPS or localhost
- Try a different browser (Chrome recommended)

### Face Detection Not Working
- Ensure good lighting
- Face should be clearly visible
- Wait for model to load (check console)

### No Audio Visualizer
- Check microphone permissions
- Verify audio is being captured
- Look for errors in console

## 📊 Performance

- **Video**: 30 FPS capture and processing
- **Audio**: 16kHz sampling with 2-second buffering
- **Face Detection**: ~30ms per frame (BlazeFace)
- **Network**: Optimized binary WebSocket messages

## 🔐 Security

- All processing happens client-side
- No data stored on frontend
- WebSocket connections use ws:// (use wss:// for production)
- Camera/microphone require explicit user permission

## 📱 Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome  | ✅ Full | Recommended |
| Firefox | ✅ Full | Excellent |
| Edge    | ✅ Full | Good |
| Safari  | ⚠️ Partial | Some features may vary |

## 🚀 Deployment

### Vercel (Recommended)

```bash
npm run build
vercel deploy
```

### Docker

```bash
docker build -t emotion-ai-frontend .
docker run -p 3000:3000 emotion-ai-frontend
```

## 📄 License

Part of the Multimodal Emotion Detection system.

## 🤝 Support

For issues:
1. Check browser console for errors
2. Review activity log in UI
3. Verify backend is running and accessible
4. Check network tab for WebSocket messages

---

**Built with Next.js 14, TypeScript, TensorFlow.js, and Tailwind CSS**
