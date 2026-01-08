"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import Webcam from "react-webcam";
import {
    LineChart, Line, CartesianGrid,
    Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell
} from "recharts";
import { Download, Trash2, TrendingUp, Info, Mic, Type, Camera, Brain, Activity, BarChart2, AlertCircle, Video } from "lucide-react";

// Define outside to prevent re-declaration on every render
const AudioVisualizer = ({ isRecording, analyser }) => {
    const [bars, setBars] = useState(new Array(32).fill(10));
    const animationFrameRef = useRef(null);

    useEffect(() => {
        if (!isRecording || !analyser) return;

        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        const update = () => {
            if (!analyser) return;
            analyser.getByteFrequencyData(dataArray);
            const normalizedData = Array.from(dataArray.slice(0, 32)).map(v => Math.max(10, v / 2.5));
            setBars(normalizedData);
            animationFrameRef.current = requestAnimationFrame(update);
        };
        animationFrameRef.current = requestAnimationFrame(update);
        return () => {
            if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        };
    }, [isRecording, analyser]);

    return (
        <div className="flex items-end justify-center space-x-1 h-32 w-full max-w-md mx-auto">
            {bars.map((h, i) => (
                <motion.div
                    key={i}
                    animate={{ height: `${h}%` }}
                    transition={{ type: "spring", stiffness: 300, damping: 20 }}
                    className="w-2 bg-gradient-to-t from-primary/20 to-primary rounded-full"
                />
            ))}
        </div>
    );
};

const EMOTION_COLORS = {
    Neutral: "#94a3b8",
    Happy: "#fbbf24",
    Sad: "#60a5fa",
    Angry: "#ef4444",
    Fear: "#a855f7",
    Disgust: "#22c55e",
    Surprise: "#ec4899"
};

const API_URL = "http://127.0.0.1:8000";

export default function EmotionDashboard() {
    const [activeTab, setActiveTab] = useState("text");
    const [text, setText] = useState("");
    const [isRecording, setIsRecording] = useState(false);
    const [isVideoActive, setIsVideoActive] = useState(false);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [history, setHistory] = useState([]);
    const [backendStatus, setBackendStatus] = useState("checking");

    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const webcamRef = useRef(null);
    const analysisIntervalRef = useRef(null);
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const audioStreamRef = useRef(null);

    // Backend status check
    const checkBackend = async () => {
        try {
            await axios.get(`${API_URL}/`);
            setBackendStatus("online");
        } catch (e) {
            setBackendStatus("offline");
        }
    };

    useEffect(() => {
        checkBackend();
        const timer = setInterval(checkBackend, 10000); // Check every 10s
        return () => clearInterval(timer);
    }, []);

    // Persistence logic
    useEffect(() => {
        const saved = localStorage.getItem("emotion_history");
        if (saved) {
            try {
                setHistory(JSON.parse(saved));
            } catch (e) {
                console.error("Failed to parse history", e);
            }
        }
    }, []);

    useEffect(() => {
        localStorage.setItem("emotion_history", JSON.stringify(history.slice(0, 20))); // Keep last 20
    }, [history]);

    const addToHistory = (result) => {
        const newEntry = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            raw_timestamp: new Date().toISOString(),
            emotion: result.dominant,
            confidence: Math.max(result.text_confidence || 0, result.audio_confidence || 0, result.vision_confidence || 0),
            modalities: [
                result.text_confidence > 0.1 ? 'Text' : '',
                result.audio_confidence > 0.1 ? 'Audio' : '',
                result.vision_confidence > 0.1 ? 'Vision' : ''
            ].filter(Boolean).join(" + "),
            all_probs: result.fused
        };
        setHistory(prev => [newEntry, ...prev]);
    };



    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioStreamRef.current = stream;

            // Setup Visualizer
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) {
                throw new Error("AudioContext not supported");
            }

            const audioCtx = new AudioContextClass();
            const analyser = audioCtx.createAnalyser();
            const source = audioCtx.createMediaStreamSource(stream);
            source.connect(analyser);
            analyser.fftSize = 256;

            audioContextRef.current = audioCtx;
            analyserRef.current = analyser;

            mediaRecorderRef.current = new MediaRecorder(stream);
            audioChunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunksRef.current.push(event.data);
            };

            mediaRecorderRef.current.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
                await handleMultimodalUpload(audioBlob, null);

                if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
                    audioContextRef.current.close();
                }
                if (audioStreamRef.current) {
                    audioStreamRef.current.getTracks().forEach(track => track.stop());
                }
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (err) {
            console.error("Recording start failed", err);
            setError("Microphone access denied or not available.");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            if (audioStreamRef.current) {
                audioStreamRef.current.getTracks().forEach(track => track.stop());
            }
        }
    };

    const captureFrame = useCallback(() => {
        if (webcamRef.current) {
            const imageSrc = webcamRef.current.getScreenshot();
            return imageSrc;
        }
        return null;
    }, [webcamRef]);

    const handleTextSubmit = async () => {
        if (!text.trim()) return;
        setLoading(true);
        setError(null);
        try {
            const resp = await axios.post(`${API_URL}/predict/text`, { text });
            setResults(resp.data);
            addToHistory(resp.data);
        } catch (err) {
            setError("Failed to analyze text. Is the backend running?");
        } finally {
            setLoading(false);
        }
    };

    const handleMultimodalUpload = async (audioBlob, imageBase64) => {
        setLoading(true);
        setError(null);
        const formData = new FormData();

        if (text) formData.append("text", text);
        if (audioBlob) formData.append("audio", audioBlob, "audio.wav");

        if (imageBase64) {
            const res = await fetch(imageBase64);
            const blob = await res.blob();
            formData.append("video", blob, "frame.jpg");
        }

        try {
            const resp = await axios.post(`${API_URL}/predict/multimodal`, formData);
            setResults(resp.data);
            addToHistory(resp.data);
            if (error) setError(null);
        } catch (err) {
            console.error("Multimodal error:", err);
            const msg = err.response ? `Analysis failed: ${err.response.data.detail || 'Server error'}` : "Cannot connect to backend server. Is it running?";
            setError(msg);
        } finally {
            setLoading(false);
        }
    };

    const analyzeVideoFrame = async () => {
        if (!isVideoActive || activeTab !== "video") return;

        const frame = captureFrame();
        if (frame) {
            try {
                const res = await fetch(frame);
                const blob = await res.blob();
                const formData = new FormData();
                formData.append("file", blob, "frame.jpg");
                const resp = await axios.post(`${API_URL}/predict/vision`, formData);
                setResults(resp.data);
                if (error && error.includes("backend")) setError(null); // Clear error if it was a connection issue
            } catch (err) {
                // Background analysis failed
                console.warn("Background frame analysis failed:", err.message);
                if (!err.response) {
                    setError("Backend server is offline or unreachable. Please start the backend.");
                } else if (err.response.status === 413) {
                    setError("Image frame is too large for the backend.");
                } else {
                    console.error("Vision API Error:", err.response.data);
                }
            }
        }
    };

    useEffect(() => {
        if (isVideoActive && activeTab === "video") {
            analysisIntervalRef.current = setInterval(analyzeVideoFrame, 3000);
        } else {
            if (analysisIntervalRef.current) clearInterval(analysisIntervalRef.current);
        }
        return () => {
            if (analysisIntervalRef.current) clearInterval(analysisIntervalRef.current);
        };
    }, [isVideoActive, activeTab]);

    const generateReport = async () => {
        if (history.length === 0) return;

        setLoading(true);
        try {
            // Lazy load jsPDF to avoid SSR issues
            const { default: jsPDF } = await import("jspdf");
            await import("jspdf-autotable");

            const doc = jsPDF();
            const timestamp = new Date().toLocaleString();

            // Header
            doc.setFontSize(22);
            doc.setTextColor(30, 27, 75); // Dark blue
            doc.text("Emotional Intelligence Report", 14, 22);

            doc.setFontSize(10);
            doc.setTextColor(100);
            doc.text(`Generated on: ${timestamp}`, 14, 30);

            // Summary Table
            const tableData = [...history].reverse().map(item => [
                item.timestamp,
                item.emotion,
                `${(item.confidence * 100).toFixed(1)}%`,
                item.modalities
            ]);

            doc.autoTable({
                head: [['Time', 'Dominant Emotion', 'Confidence', 'Modalities']],
                body: tableData,
                startY: 40,
                theme: 'grid',
                headStyles: { fillColor: [139, 92, 246] }, // Primary color
            });

            doc.save(`Emotion_Report_${Date.now()}.pdf`);
        } catch (err) {
            console.error("PDF generation failed", err);
            setError("Failed to generate PDF report.");
        } finally {
            setLoading(false);
        }
    };

    const chartData = results ? Object.entries(results.fused).map(([name, value]) => ({
        name,
        value: parseFloat((value * 100).toFixed(1)),
        fullMark: 100
    })) : [];

    const trendData = [...history].reverse().map(item => ({
        time: item.timestamp,
        confidence: parseFloat((item.confidence * 100).toFixed(1)),
        emotion: item.emotion
    }));



    return (
        <div className="w-full max-w-6xl mx-auto p-4 space-y-8">
            {/* Input Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="glass rounded-3xl p-8 space-y-6"
                >
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-3">
                            <div className="bg-primary/20 p-2 rounded-lg">
                                <Activity className="text-primary w-6 h-6" />
                            </div>
                            <h2 className="text-2xl font-bold tracking-tight">Emotional Input</h2>
                        </div>
                        <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-white/5 border border-white/10">
                            <div className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-green-500 animate-pulse' : backendStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'}`} />
                            <span className="text-[10px] uppercase font-bold text-white/40">Backend: {backendStatus}</span>
                        </div>
                    </div>

                    <div className="flex space-x-2 p-1 bg-white/5 rounded-xl">
                        {[
                            { id: "text", icon: Type, label: "Text" },
                            { id: "audio", icon: Mic, label: "Audio" },
                            { id: "video", icon: Video, label: "Video" }
                        ].map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex-1 flex items-center justify-center py-2 rounded-lg transition-all ${activeTab === tab.id ? "bg-primary text-white shadow-lg shadow-primary/25" : "hover:bg-white/5 text-white/50"}`}
                            >
                                <tab.icon className="w-4 h-4 mr-2" /> {tab.label}
                            </button>
                        ))}
                    </div>

                    <AnimatePresence mode="wait">
                        {activeTab === "text" && (
                            <motion.div
                                key="text"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                className="space-y-4"
                            >
                                <textarea
                                    value={text}
                                    onChange={(e) => setText(e.target.value)}
                                    placeholder="How are you feeling? Type your thoughts here..."
                                    className="w-full h-40 bg-white/5 rounded-2xl p-4 focus:ring-2 focus:ring-primary outline-none resize-none transition-all placeholder:text-white/30 text-white"
                                />
                                <button
                                    onClick={handleTextSubmit}
                                    disabled={loading || !text}
                                    className="w-full py-4 bg-gradient-to-r from-primary to-secondary rounded-xl font-bold hover:opacity-90 disabled:opacity-50 transition-all flex items-center justify-center text-white"
                                >
                                    {loading ? <div className="animate-spin rounded-full h-5 w-5 border-2 border-white/20 border-t-white" /> : "Analyze Sentiment"}
                                </button>
                            </motion.div>
                        )}

                        {activeTab === "audio" && (
                            <motion.div
                                key="audio"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                className="space-y-6 flex flex-col items-center justify-center py-8"
                            >
                                {isRecording ? (
                                    <AudioVisualizer isRecording={isRecording} analyser={analyserRef.current} />
                                ) : (
                                    <div className="w-32 h-32 rounded-full border-2 border-dashed border-white/10 flex items-center justify-center">
                                        <Mic className="text-white/10 w-12 h-12" />
                                    </div>
                                )}

                                <div className={`relative ${isRecording ? 'animate-pulse' : ''}`}>
                                    <button
                                        onClick={isRecording ? stopRecording : startRecording}
                                        className={`w-32 h-32 rounded-full flex items-center justify-center transition-all ${isRecording ? 'bg-red-500 shadow-lg shadow-red-500/50' : 'bg-primary/20 hover:bg-primary/30 border-2 border-primary'}`}
                                    >
                                        <Mic className={`w-12 h-12 ${isRecording ? 'text-white' : 'text-primary'}`} />
                                    </button>
                                </div>
                                <p className="text-white/60 font-medium tracking-wide">
                                    {isRecording ? "Listening to your voice..." : "Click to record audio"}
                                </p>
                            </motion.div>
                        )}

                        {activeTab === "video" && (
                            <motion.div
                                key="video"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -10 }}
                                className="space-y-4"
                            >
                                <div className="relative rounded-2xl overflow-hidden glass border-white/10 aspect-video flex items-center justify-center bg-black/40">
                                    {isVideoActive ? (
                                        <Webcam
                                            audio={false}
                                            ref={webcamRef}
                                            screenshotFormat="image/jpeg"
                                            className="w-full h-full object-cover"
                                        />
                                    ) : (
                                        <div className="flex flex-col items-center text-white/20">
                                            <Camera className="w-16 h-16 mb-2" />
                                            <p>Camera is inactive</p>
                                        </div>
                                    )}
                                </div>
                                <button
                                    onClick={() => setIsVideoActive(!isVideoActive)}
                                    className={`w-full py-3 rounded-xl font-bold transition-all flex items-center justify-center ${isVideoActive ? 'bg-red-500/20 text-red-500 border border-red-500/50' : 'bg-accent text-white'}`}
                                >
                                    {isVideoActive ? "Stop Camera" : "Start Real-time Analysis"}
                                </button>
                                {isVideoActive && (
                                    <button
                                        onClick={() => handleMultimodalUpload(null, captureFrame())}
                                        className="w-full py-3 bg-white/5 hover:bg-white/10 text-white rounded-xl border border-white/10 transition-all font-medium"
                                    >
                                        Capture & Analyze Tri-Modal
                                    </button>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {error && (
                        <div className="flex items-center p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm">
                            <AlertCircle className="w-4 h-4 mr-2" />
                            {error}
                        </div>
                    )}
                </motion.div>

                {/* Results Section */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="glass rounded-3xl p-8 flex flex-col"
                >
                    <div className="flex items-center justify-between mb-8">
                        <div className="flex items-center space-x-3">
                            <div className="bg-secondary/20 p-2 rounded-lg">
                                <BarChart2 className="text-secondary w-6 h-6" />
                            </div>
                            <h2 className="text-2xl font-bold tracking-tight">Detection Results</h2>
                        </div>
                        {results && (
                            <div className="px-4 py-1.5 bg-secondary/10 border border-secondary/20 rounded-full text-secondary text-sm font-bold uppercase tracking-wider">
                                {results.dominant}
                            </div>
                        )}
                    </div>

                    {!results && !loading && (
                        <div className="flex-1 flex flex-col items-center justify-center text-white/20 space-y-4">
                            <Brain className="w-20 h-20 opacity-10" />
                            <p className="text-center font-medium">Capture your feelings to see <br />AI-powered analysis</p>
                        </div>
                    )}

                    {loading && (
                        <div className="flex-1 flex flex-col items-center justify-center space-y-6">
                            <div className="relative w-24 h-24">
                                <div className="absolute inset-0 border-4 border-primary/20 rounded-full" />
                                <div className="absolute inset-0 border-4 border-t-primary rounded-full animate-spin" />
                            </div>
                            <p className="text-primary animate-pulse-slow font-bold tracking-widest uppercase text-sm">Processing Neural Fusion</p>
                        </div>
                    )}

                    {results && !loading && (
                        <div className="flex-1 flex flex-col space-y-8">
                            <div className="h-[300px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                                        <PolarGrid stroke="#ffffff20" />
                                        <PolarAngleAxis dataKey="name" tick={{ fill: '#ffffff60', fontSize: 12 }} />
                                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                        <Radar
                                            name="Emotion"
                                            dataKey="value"
                                            stroke="#8b5cf6"
                                            fill="#8b5cf6"
                                            fillOpacity={0.4}
                                        />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1e1b4b', border: 'none', borderRadius: '12px', color: '#fff' }}
                                        />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </div>

                            <div className="grid grid-cols-3 gap-2">
                                {[
                                    { label: "Text", val: results.text_confidence, color: "bg-primary" },
                                    { label: "Audio", val: results.audio_confidence, color: "bg-secondary" },
                                    { label: "Vision", val: results.vision_confidence, color: "bg-accent" }
                                ].map((item) => (
                                    <div key={item.label} className="p-3 bg-white/5 rounded-2xl border border-white/10">
                                        <p className="text-white/40 text-[10px] uppercase font-bold mb-1">{item.label}</p>
                                        <p className="text-lg font-bold">{(item.val * 100).toFixed(0)}%</p>
                                        <div className="w-full bg-white/10 h-1 rounded-full mt-2 overflow-hidden">
                                            <div className={`${item.color} h-full`} style={{ width: `${item.val * 100}%` }} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </motion.div>
            </div>

            {/* Probability Bars and History */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass rounded-3xl p-8 lg:col-span-2 space-y-8"
                >
                    <div>
                        <div className="flex items-center space-x-2 mb-6 text-white/60">
                            <TrendingUp className="w-4 h-4" />
                            <h3 className="text-lg font-bold">Emotion Probabilities</h3>
                        </div>
                        <div className="h-[200px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} layout="vertical">
                                    <XAxis type="number" hide domain={[0, 100]} />
                                    <YAxis dataKey="name" type="category" width={80} tick={{ fill: '#ffffff60', fontSize: 12 }} axisLine={false} tickLine={false} />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                        {chartData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={EMOTION_COLORS[entry.name] || '#8b5cf6'} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {history.length > 2 && (
                        <div className="pt-8 border-t border-white/5">
                            <div className="flex items-center space-x-2 mb-6 text-white/60">
                                <Activity className="w-4 h-4" />
                                <h3 className="text-lg font-bold">Session Confidence Trend</h3>
                            </div>
                            <div className="h-[150px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={trendData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" />
                                        <XAxis dataKey="time" hide />
                                        <YAxis domain={[0, 100]} hide />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1e1b4b', border: 'none', borderRadius: '12px', color: '#fff' }}
                                        />
                                        <Line type="monotone" dataKey="confidence" stroke="#8b5cf6" strokeWidth={3} dot={{ fill: '#8b5cf6', r: 4 }} activeDot={{ r: 6 }} />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    )}
                </motion.div>

                {/* Session History */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass rounded-3xl p-8 overflow-hidden flex flex-col"
                >
                    <div className="flex items-center justify-between mb-6">
                        <h3 className="text-lg font-bold text-white/60">Session History</h3>
                        <div className="flex items-center space-x-2">
                            {history.length > 0 && (
                                <button
                                    onClick={generateReport}
                                    title="Download PDF Report"
                                    className="p-2 hover:bg-white/5 rounded-lg text-primary transition-colors"
                                >
                                    <Download className="w-4 h-4" />
                                </button>
                            )}
                            <button
                                onClick={() => setHistory([])}
                                title="Clear History"
                                className="p-2 hover:bg-white/5 rounded-lg text-white/30 hover:text-red-400 transition-colors"
                            >
                                <Trash2 className="w-4 h-4" />
                            </button>
                        </div>
                    </div>

                    <div className="space-y-4 overflow-y-auto max-h-[400px] pr-2 custom-scrollbar flex-1">
                        {history.length === 0 ? (
                            <div className="flex flex-col items-center justify-center py-12 text-white/10">
                                <Info className="w-8 h-8 mb-2" />
                                <p className="text-sm italic">No records yet</p>
                            </div>
                        ) : (
                            history.map((item) => (
                                <motion.div
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    key={item.id}
                                    className="p-3 bg-white/5 rounded-xl border border-white/5 flex items-center justify-between group hover:bg-white/10 transition-all cursor-default"
                                >
                                    <div>
                                        <p className="text-xs text-white/40">{item.timestamp}</p>
                                        <p className="font-bold text-sm" style={{ color: EMOTION_COLORS[item.emotion] }}>{item.emotion}</p>
                                    </div>
                                    <div className="text-right">
                                        <p className="text-[10px] text-white/30 uppercase font-bold">{item.modalities}</p>
                                        <p className="text-xs font-mono">{(item.confidence * 100).toFixed(0)}%</p>
                                    </div>
                                </motion.div>
                            ))
                        )}
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
