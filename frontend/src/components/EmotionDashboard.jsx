"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import Webcam from "react-webcam";
import {
    Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
    ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
    LineChart, Line, CartesianGrid
} from "recharts";
import {
    Download, Trash2, TrendingUp, Info, Camera, Brain,
    Activity, BarChart2, AlertCircle, Video, Upload,
    Play, Pause, RefreshCcw, LogIn, Github
} from "lucide-react";

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
const WS_URL = "ws://127.0.0.1:8000/ws/stream";

export default function EmotionDashboard() {
    const [activeMode, setActiveMode] = useState("stream"); // "stream" or "upload"
    const [isStreaming, setIsStreaming] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [history, setHistory] = useState([]);
    const [backendStatus, setBackendStatus] = useState("checking");
    const [uploadResults, setUploadResults] = useState(null);

    const webcamRef = useRef(null);
    const socketRef = useRef(null);
    const streamIntervalRef = useRef(null);

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
        const timer = setInterval(checkBackend, 10000);
        return () => clearInterval(timer);
    }, []);

    // WebSocket Logic for Streaming
    const startStreaming = useCallback(() => {
        if (socketRef.current) return;

        socketRef.current = new WebSocket(WS_URL);

        socketRef.current.onopen = () => {
            console.log("WebSocket Connected");
            setIsStreaming(true);
            setError(null);

            // Start sending frames
            streamIntervalRef.current = setInterval(() => {
                if (webcamRef.current && socketRef.current?.readyState === WebSocket.OPEN) {
                    const screenshot = webcamRef.current.getScreenshot();
                    if (screenshot) {
                        socketRef.current.send(screenshot);
                    }
                }
            }, 1000); // 1 frame per second
        };

        socketRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.error) {
                setError(data.error);
            } else {
                setResults(data);
                addToHistory(data);
            }
        };

        socketRef.current.onclose = () => {
            console.log("WebSocket Disconnected");
            stopStreaming();
        };

        socketRef.current.onerror = (err) => {
            console.error("WebSocket Error:", err);
            setError("WebSocket connection failed.");
            stopStreaming();
        };
    }, []);

    const stopStreaming = useCallback(() => {
        setIsStreaming(false);
        if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
        if (socketRef.current) {
            socketRef.current.close();
            socketRef.current = null;
        }
    }, []);

    useEffect(() => {
        return () => stopStreaming();
    }, [stopStreaming]);

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setIsUploading(true);
        setError(null);
        setUploadProgress(10);

        const formData = new FormData();
        formData.append("file", file);

        try {
            setUploadProgress(30);
            const resp = await axios.post(`${API_URL}/predict/video-upload`, formData, {
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setUploadProgress(30 + (percentCompleted * 0.7)); // Scale progress to remaining 70%
                }
            });
            setUploadResults(resp.data);
            setResults(resp.data.summary);
            setUploadProgress(100);
            setTimeout(() => setIsUploading(false), 500);
        } catch (err) {
            console.error("Upload error:", err);
            setError("Failed to process video upload. Make sure it's a valid video file.");
            setIsUploading(false);
        }
    };

    const addToHistory = (result) => {
        const newEntry = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            emotion: result.dominant,
            confidence: result.vision_confidence || 0,
            probs: result.fused
        };
        setHistory(prev => [newEntry, ...prev].slice(0, 20));
    };

    const chartData = results ? Object.entries(results.fused || results.average_probs).map(([name, value]) => ({
        name,
        value: parseFloat((value * 100).toFixed(1)),
        fullMark: 100
    })) : [];

    const timelineData = uploadResults?.timeline ? uploadResults.timeline.map(t => ({
        time: t.second,
        ...Object.fromEntries(Object.entries(t.probs).map(([k, v]) => [k, (v * 100).toFixed(1)]))
    })) : [];

    return (
        <div className="w-full max-w-7xl mx-auto p-4 space-y-8 min-h-screen bg-slate-950 text-white font-sans">
            {/* Header / OAuth Section */}
            <div className="flex flex-col md:flex-row justify-between items-center bg-white/5 p-6 rounded-3xl border border-white/10 glass mb-8">
                <div className="flex items-center space-x-4 mb-4 md:mb-0">
                    <div className="bg-primary p-3 rounded-2xl shadow-lg shadow-primary/20">
                        <Brain className="w-8 h-8 text-white" />
                    </div>
                    <div>
                        <h1 className="text-3xl font-black bg-gradient-to-r from-white to-white/40 bg-clip-text text-transparent italic tracking-tighter uppercase">
                            Sentience AI
                        </h1>
                        <p className="text-xs font-bold text-primary tracking-widest uppercase">Video Emotion Analytics</p>
                    </div>
                </div>

                <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-white/5 border border-white/10">
                        <div className={`w-2 h-2 rounded-full ${backendStatus === 'online' ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                        <span className="text-[10px] uppercase font-black text-white/60 tracking-widest">
                            API: {backendStatus}
                        </span>
                    </div>
                    <button className="flex items-center space-x-2 px-4 py-2 bg-white text-black rounded-xl font-bold hover:bg-white/90 transition-all text-sm">
                        <LogIn className="w-4 h-4" />
                        <span>Sign In</span>
                    </button>
                    <button className="p-2 bg-white/5 border border-white/10 rounded-xl hover:bg-white/10 transition-all">
                        <Github className="w-5 h-5 text-white/60" />
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left Column: Input (Streaming/Upload) */}
                <div className="lg:col-span-12 xl:col-span-7 space-y-8">
                    <div className="glass rounded-[2rem] overflow-hidden border border-white/5 shadow-2xl">
                        <div className="flex border-b border-white/5">
                            <button
                                onClick={() => { setActiveMode("stream"); stopStreaming(); setResults(null); }}
                                className={`flex-1 py-6 font-black uppercase tracking-widest text-sm transition-all flex items-center justify-center space-x-3 ${activeMode === "stream" ? "bg-white/10 text-white" : "text-white/30 hover:text-white/50"}`}
                            >
                                <Video className="w-5 h-5" />
                                <span>Live Stream</span>
                            </button>
                            <button
                                onClick={() => { setActiveMode("upload"); stopStreaming(); setResults(null); }}
                                className={`flex-1 py-6 font-black uppercase tracking-widest text-sm transition-all flex items-center justify-center space-x-3 ${activeMode === "upload" ? "bg-white/10 text-white" : "text-white/30 hover:text-white/50"}`}
                            >
                                <Upload className="w-5 h-5" />
                                <span>Video Upload</span>
                            </button>
                        </div>

                        <div className="p-8">
                            <AnimatePresence mode="wait">
                                {activeMode === "stream" ? (
                                    <motion.div
                                        key="stream"
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        exit={{ opacity: 0, y: -20 }}
                                        className="space-y-6"
                                    >
                                        <div className="relative rounded-3xl overflow-hidden aspect-video bg-black/40 border-4 border-white/5 group shadow-inner">
                                            {isStreaming ? (
                                                <>
                                                    <Webcam
                                                        audio={false}
                                                        ref={webcamRef}
                                                        screenshotFormat="image/jpeg"
                                                        className="w-full h-full object-cover scale-105"
                                                    />
                                                    <div className="absolute top-6 left-6 flex items-center space-x-2 bg-red-600 px-3 py-1 rounded-full animate-pulse">
                                                        <div className="w-2 h-2 bg-white rounded-full" />
                                                        <span className="text-[10px] font-black uppercase tracking-tighter">Live</span>
                                                    </div>
                                                </>
                                            ) : (
                                                <div className="absolute inset-0 flex flex-col items-center justify-center space-y-4">
                                                    <div className="p-6 rounded-full bg-white/5 border border-white/10 group-hover:scale-110 transition-transform duration-500">
                                                        <Camera className="w-16 h-16 text-white/10" />
                                                    </div>
                                                    <p className="text-white/20 font-bold tracking-widest uppercase text-xs">Camera Inactive</p>
                                                </div>
                                            )}
                                        </div>

                                        <button
                                            onClick={isStreaming ? stopStreaming : startStreaming}
                                            className={`w-full py-6 rounded-2xl font-black uppercase tracking-widest transition-all shadow-xl flex items-center justify-center space-x-4 ${isStreaming ? 'bg-red-500/20 text-red-500 border border-red-500/50 hover:bg-red-500/30' : 'bg-primary text-white hover:opacity-90 active:scale-[0.98]'}`}
                                        >
                                            {isStreaming ? <><Pause className="w-6 h-6" /> <span>Stop Analysis</span></> : <><Play className="w-6 h-6" /> <span>Start Live Capture</span></>}
                                        </button>
                                    </motion.div>
                                ) : (
                                    <motion.div
                                        key="upload"
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        exit={{ opacity: 0, y: -20 }}
                                        className="space-y-6"
                                    >
                                        <div className="relative rounded-3xl border-2 border-dashed border-white/10 h-[400px] flex flex-col items-center justify-center space-y-6 bg-white/[0.02] hover:bg-white/[0.04] transition-all cursor-pointer group" onClick={() => document.getElementById('video-upload').click()}>
                                            <input type="file" id="video-upload" className="hidden" accept="video/*" onChange={handleFileUpload} />
                                            {isUploading ? (
                                                <div className="flex flex-col items-center space-y-6 w-full max-w-xs">
                                                    <div className="relative w-24 h-24">
                                                        <div className="absolute inset-0 border-4 border-white/10 rounded-full" />
                                                        <div className="absolute inset-0 border-4 border-t-primary rounded-full animate-spin" />
                                                    </div>
                                                    <p className="text-primary font-black uppercase tracking-widest text-xs animate-pulse">Neural Processing...</p>
                                                    <div className="w-full bg-white/10 h-1.5 rounded-full overflow-hidden">
                                                        <motion.div
                                                            className="bg-primary h-full"
                                                            initial={{ width: 0 }}
                                                            animate={{ width: `${uploadProgress}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            ) : (
                                                <>
                                                    <div className="p-8 rounded-full bg-white/5 border border-white/10 group-hover:border-primary/50 transition-colors duration-500">
                                                        <Upload className="w-16 h-16 text-white/20 group-hover:text-primary transition-colors duration-500" />
                                                    </div>
                                                    <div className="text-center">
                                                        <p className="text-lg font-black uppercase tracking-tighter italic">Drop Video Here</p>
                                                        <p className="text-white/30 text-xs font-bold uppercase tracking-widest mt-2">MP4, WEBM, MOV supported</p>
                                                    </div>
                                                </>
                                            )}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>

                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    className="mt-6 flex items-center p-5 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-400 text-sm font-bold shadow-lg"
                                >
                                    <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0" />
                                    {error}
                                </motion.div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Right Column: Analytics */}
                <div className="lg:col-span-12 xl:col-span-5 space-y-8">
                    <div className="glass rounded-[2rem] p-10 border border-white/5 flex flex-col min-h-[500px] shadow-2xl">
                        <div className="flex items-center justify-between mb-10">
                            <div className="flex items-center space-x-3">
                                <div className="bg-secondary/20 p-2 rounded-xl">
                                    <BarChart2 className="text-secondary w-6 h-6" />
                                </div>
                                <h2 className="text-xl font-black italic uppercase tracking-tighter">Real-time Insights</h2>
                            </div>
                            {results && (
                                <div className="px-5 py-2 bg-primary/10 border border-primary/20 rounded-full text-primary text-xs font-black uppercase tracking-widest shadow-lg">
                                    {results.dominant}
                                </div>
                            )}
                        </div>

                        {!results && !isUploading && (
                            <div className="flex-1 flex flex-col items-center justify-center text-white/5 space-y-6">
                                <Brain className="w-32 h-32 opacity-20" />
                                <div className="text-center">
                                    <p className="text-lg font-black italic uppercase tracking-tighter">Awaiting Signal</p>
                                    <p className="text-xs font-bold uppercase tracking-widest opacity-40 mt-1">Initialize stream to begin analysis</p>
                                </div>
                            </div>
                        )}

                        {results && (
                            <div className="flex-1 space-y-12">
                                <div className="h-[280px] w-full">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                                            <PolarGrid stroke="#ffffff10" />
                                            <PolarAngleAxis dataKey="name" tick={{ fill: '#ffffff40', fontSize: 10, fontWeight: 800 }} />
                                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                            <Radar
                                                name="Emotion"
                                                dataKey="value"
                                                stroke="#8b5cf6"
                                                fill="#8b5cf6"
                                                fillOpacity={0.4}
                                            />
                                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: 'none', borderRadius: '16px', color: '#fff' }} />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>

                                <div className="grid grid-cols-1 gap-4">
                                    {chartData.sort((a, b) => b.value - a.value).slice(0, 3).map((item) => (
                                        <div key={item.name} className="p-5 bg-white/[0.03] rounded-2xl border border-white/5 hover:bg-white/[0.05] transition-all">
                                            <div className="flex justify-between items-end mb-3">
                                                <p className="text-white/40 text-[10px] font-black uppercase tracking-widest">{item.name}</p>
                                                <p className="text-sm font-black italic">{item.value}%</p>
                                            </div>
                                            <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden">
                                                <motion.div
                                                    className="h-full bg-gradient-to-r from-primary to-secondary"
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${item.value}%` }}
                                                    transition={{ duration: 0.8, ease: "easeOut" }}
                                                />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Timeline for Uploads */}
            {activeMode === "upload" && uploadResults && (
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass rounded-[2rem] p-10 border border-white/5 shadow-2xl"
                >
                    <div className="flex items-center space-x-3 mb-10 text-white/40">
                        <TrendingUp className="w-5 h-5" />
                        <h3 className="text-lg font-black italic uppercase tracking-tighter">Neural Timeline</h3>
                    </div>
                    <div className="h-[250px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={timelineData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fill: '#ffffff20', fontSize: 10 }} label={{ value: 'Seconds', position: 'bottom', fill: '#ffffff20' }} />
                                <YAxis domain={[0, 100]} axisLine={false} tickLine={false} tick={{ fill: '#ffffff20', fontSize: 10 }} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#0f172a', border: 'none', borderRadius: '16px', color: '#fff', fontSize: '10px' }}
                                />
                                {Object.keys(EMOTION_COLORS).map(emotion => (
                                    <Line
                                        key={emotion}
                                        type="monotone"
                                        dataKey={emotion}
                                        stroke={EMOTION_COLORS[emotion]}
                                        strokeWidth={results?.dominant === emotion ? 4 : 2}
                                        dot={false}
                                        opacity={results?.dominant === emotion ? 1 : 0.3}
                                    />
                                ))}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </motion.div>
            )}

            {/* Global History Overlay (Optional footer or sidebar) */}
            {history.length > 0 && activeMode === "stream" && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {history.slice(0, 4).map((item) => (
                        <motion.div
                            key={item.id}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="bg-white/5 border border-white/5 p-5 rounded-2xl flex items-center justify-between"
                        >
                            <div>
                                <p className="text-[10px] text-white/20 font-black uppercase tracking-widest">{item.timestamp}</p>
                                <p className="font-black italic uppercase tracking-tighter" style={{ color: EMOTION_COLORS[item.emotion] }}>{item.emotion}</p>
                            </div>
                            <div className="text-right">
                                <p className="text-[10px] text-white/20 font-black uppercase tracking-widest">Confidence</p>
                                <p className="text-sm font-mono tracking-tighter">{(item.confidence * 100).toFixed(0)}%</p>
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}
        </div>
    );
}
