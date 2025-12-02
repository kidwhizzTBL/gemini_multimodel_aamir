import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, LiveServerMessage, Modality, Type, GenerateContentResponse } from '@google/genai';
import { Mic, MicOff, Video, MessageSquare, Play, Square, Send, Image as ImageIcon, Loader2, Download, AlertCircle } from 'lucide-react';

// --- Types & Interfaces ---

type Tab = 'live' | 'veo' | 'chat';

interface Message {
  role: 'user' | 'model';
  text: string;
  image?: string;
}

// --- Helper Functions for Audio/Live API ---

function base64ToBytes(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const length = binaryString.length;
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// --- Components ---

const Header = ({ activeTab, onTabChange }: { activeTab: Tab; onTabChange: (t: Tab) => void }) => (
  <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md z-10">
    <div className="flex items-center gap-2">
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
        <span className="font-bold text-white">G</span>
      </div>
      <h1 className="text-xl font-semibold tracking-tight">Gemini Studio</h1>
    </div>
    <nav className="flex items-center gap-1 bg-slate-800/50 p-1 rounded-xl">
      <button
        onClick={() => onTabChange('live')}
        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
          activeTab === 'live' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
        }`}
      >
        <div className="flex items-center gap-2">
          <Mic className="w-4 h-4" />
          Live
        </div>
      </button>
      <button
        onClick={() => onTabChange('veo')}
        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
          activeTab === 'veo' ? 'bg-purple-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
        }`}
      >
        <div className="flex items-center gap-2">
          <Video className="w-4 h-4" />
          Veo
        </div>
      </button>
      <button
        onClick={() => onTabChange('chat')}
        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
          activeTab === 'chat' ? 'bg-emerald-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
        }`}
      >
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4" />
          Chat
        </div>
      </button>
    </nav>
  </header>
);

// --- Live Conversation Component ---

const LiveView = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [volume, setVolume] = useState(0);

  // Audio Context Refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const sessionRef = useRef<any>(null); // To hold the session object for sending data
  const frameIntervalRef = useRef<number | null>(null);

  const cleanup = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    
    // Stop audio sources
    sourcesRef.current.forEach(source => {
      try { source.stop(); } catch(e) {}
    });
    sourcesRef.current.clear();

    // Close contexts
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (inputAudioContextRef.current) {
      inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }

    // Stop tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    // Close session
    if (sessionRef.current) {
       // We can't explicitly close the session object from the SDK easily if it's not exposed, 
       // but we can stop sending data. 
       // In a real app, we'd signal the close.
       sessionRef.current = null;
    }
    
    setIsConnected(false);
    setVolume(0);
  }, []);

  const connect = async () => {
    setError(null);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      // Initialize Audio Contexts
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      
      // Get User Media
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: { width: 640, height: 480 } });
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      const config = {
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
          },
          systemInstruction: "You are a helpful, witty, and concise AI assistant. You can see the user via video.",
        },
      };

      // Connect to Live API
      const sessionPromise = ai.live.connect({
        ...config,
        callbacks: {
          onopen: () => {
            console.log('Live session connected');
            setIsConnected(true);
            
            // Start Audio Input Streaming
            if (!inputAudioContextRef.current || !streamRef.current) return;
            
            const source = inputAudioContextRef.current.createMediaStreamSource(streamRef.current);
            sourceRef.current = source;
            
            const processor = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;
            
            processor.onaudioprocess = (e) => {
              if (isMuted) return;
              const inputData = e.inputBuffer.getChannelData(0);
              
              // Simple volume meter
              let sum = 0;
              for(let i=0; i<inputData.length; i++) sum += inputData[i] * inputData[i];
              setVolume(Math.sqrt(sum / inputData.length));

              // PCM conversion to Int16
              const l = inputData.length;
              const int16 = new Int16Array(l);
              for (let i = 0; i < l; i++) {
                int16[i] = inputData[i] * 32768;
              }
              
              const base64Data = bytesToBase64(new Uint8Array(int16.buffer));
              
              sessionPromise.then(session => {
                session.sendRealtimeInput({
                  media: {
                    mimeType: 'audio/pcm;rate=16000',
                    data: base64Data
                  }
                });
              });
            };
            
            source.connect(processor);
            processor.connect(inputAudioContextRef.current.destination);

            // Start Video Input Streaming
            if (canvasRef.current && videoRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                frameIntervalRef.current = window.setInterval(() => {
                    if (videoRef.current && ctx && !isMuted) {
                        canvasRef.current!.width = videoRef.current.videoWidth;
                        canvasRef.current!.height = videoRef.current.videoHeight;
                        ctx.drawImage(videoRef.current, 0, 0);
                        const base64Data = canvasRef.current!.toDataURL('image/jpeg', 0.5).split(',')[1];
                         sessionPromise.then(session => {
                            session.sendRealtimeInput({
                                media: { mimeType: 'image/jpeg', data: base64Data }
                            });
                        });
                    }
                }, 1000); // 1 FPS for video to save bandwidth
            }
          },
          onmessage: async (msg: LiveServerMessage) => {
            const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData && audioContextRef.current) {
              const ctx = audioContextRef.current;
              const bytes = base64ToBytes(audioData);
              
              // Decode PCM 24000Hz (Model output)
              // We need to implement manual decoding because ctx.decodeAudioData expects headers (wav/mp3)
              const dataInt16 = new Int16Array(bytes.buffer);
              const float32 = new Float32Array(dataInt16.length);
              for(let i=0; i<dataInt16.length; i++) {
                float32[i] = dataInt16[i] / 32768.0;
              }
              
              const buffer = ctx.createBuffer(1, float32.length, 24000);
              buffer.getChannelData(0).set(float32);

              const source = ctx.createBufferSource();
              source.buffer = buffer;
              source.connect(ctx.destination);
              
              // Schedule playback
              const currentTime = ctx.currentTime;
              const startTime = Math.max(currentTime, nextStartTimeRef.current);
              source.start(startTime);
              nextStartTimeRef.current = startTime + buffer.duration;
              
              sourcesRef.current.add(source);
              source.onended = () => sourcesRef.current.delete(source);
            }
            
            if (msg.serverContent?.interrupted) {
               sourcesRef.current.forEach(s => s.stop());
               sourcesRef.current.clear();
               nextStartTimeRef.current = 0;
            }
          },
          onclose: () => {
            console.log('Session closed');
            cleanup();
          },
          onerror: (err) => {
            console.error('Session error', err);
            setError("Connection error. Please try again.");
            cleanup();
          }
        }
      });
      
      sessionRef.current = sessionPromise;

    } catch (e: any) {
      console.error(e);
      setError(e.message || "Failed to connect");
      cleanup();
    }
  };

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  return (
    <div className="flex flex-col h-full p-6 max-w-4xl mx-auto w-full gap-6">
      <div className="flex-1 bg-slate-900 rounded-2xl border border-slate-800 relative overflow-hidden flex flex-col items-center justify-center shadow-2xl">
         {/* Video Feed */}
         <video 
            ref={videoRef} 
            className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-500 ${isConnected ? 'opacity-100' : 'opacity-20 grayscale'}`}
            muted 
            playsInline 
         />
         <canvas ref={canvasRef} className="hidden" />
         
         {/* Visualizer Overlay */}
         {isConnected && (
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-1 h-12">
               {[...Array(5)].map((_, i) => (
                  <div 
                    key={i} 
                    className="w-2 bg-blue-500 rounded-full transition-all duration-75"
                    style={{ height: `${Math.max(10, volume * 100 * (Math.random() + 0.5))}px`, opacity: 0.8 }}
                  />
               ))}
            </div>
         )}

         {!isConnected && !error && (
            <div className="z-10 text-center space-y-4">
                <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4 border border-slate-700">
                    <Mic className="w-8 h-8 text-slate-400" />
                </div>
                <h2 className="text-2xl font-bold text-white">Start Conversation</h2>
                <p className="text-slate-400 max-w-md">Connect to Gemini Live for a real-time, low-latency voice and video conversation.</p>
            </div>
         )}

         {error && (
            <div className="z-10 bg-red-500/10 border border-red-500/50 text-red-200 px-6 py-4 rounded-xl flex items-center gap-3">
                <AlertCircle className="w-5 h-5" />
                {error}
            </div>
         )}
      </div>

      <div className="flex justify-center gap-4">
        {!isConnected ? (
          <button 
            onClick={connect}
            className="flex items-center gap-3 bg-blue-600 hover:bg-blue-500 text-white px-8 py-4 rounded-full font-semibold text-lg shadow-lg hover:shadow-blue-500/20 transition-all active:scale-95"
          >
            <Play className="w-5 h-5 fill-current" />
            Connect Live
          </button>
        ) : (
          <>
             <button 
                onClick={() => setIsMuted(!isMuted)}
                className={`p-4 rounded-full border-2 transition-all ${isMuted ? 'bg-red-500/20 border-red-500 text-red-400' : 'bg-slate-800 border-slate-700 text-white hover:bg-slate-700'}`}
             >
                {isMuted ? <MicOff className="w-6 h-6" /> : <Mic className="w-6 h-6" />}
             </button>
             <button 
                onClick={cleanup}
                className="flex items-center gap-3 bg-red-600 hover:bg-red-500 text-white px-8 py-4 rounded-full font-semibold text-lg shadow-lg hover:shadow-red-500/20 transition-all active:scale-95"
             >
                <Square className="w-5 h-5 fill-current" />
                Disconnect
             </button>
          </>
        )}
      </div>
    </div>
  );
};

// --- Veo Video Generation Component ---

const VeoView = () => {
  const [prompt, setPrompt] = useState('A neon hologram of a cat driving at top speed');
  const [loading, setLoading] = useState(false);
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');
  const [config, setConfig] = useState({ resolution: '720p', aspectRatio: '16:9' });
  const [needsKey, setNeedsKey] = useState(false);

  useEffect(() => {
    checkKey();
  }, []);

  const checkKey = async () => {
     try {
        if ((window as any).aistudio) {
            const hasKey = await (window as any).aistudio.hasSelectedApiKey();
            setNeedsKey(!hasKey);
        }
     } catch (e) {
         console.warn("AI Studio API check failed", e);
     }
  };

  const selectKey = async () => {
      try {
          if ((window as any).aistudio) {
              await (window as any).aistudio.openSelectKey();
              // Assume success to proceed, handle race condition by just updating state
              setNeedsKey(false);
          }
      } catch (e) {
          console.error(e);
      }
  };

  const generateVideo = async () => {
    if (needsKey) {
        await selectKey();
    }
    
    // Double check key status right before call or create new AI instance
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    setLoading(true);
    setError(null);
    setVideoUri(null);
    setStatus('Initializing generation...');

    try {
        let operation = await ai.models.generateVideos({
            model: 'veo-3.1-fast-generate-preview',
            prompt: prompt,
            config: {
                numberOfVideos: 1,
                resolution: config.resolution as any,
                aspectRatio: config.aspectRatio as any,
            }
        });

        setStatus('Processing video (this may take a minute)...');
        
        while (!operation.done) {
            await new Promise(resolve => setTimeout(resolve, 5000));
            operation = await ai.operations.getVideosOperation({operation: operation});
            setStatus('Still processing...');
        }

        const uri = operation.response?.generatedVideos?.[0]?.video?.uri;
        if (uri) {
            // Append API key for playback
            const authenticatedUri = `${uri}&key=${process.env.API_KEY}`;
            setVideoUri(authenticatedUri);
        } else {
            setError('No video URI returned.');
        }

    } catch (e: any) {
        console.error(e);
        if (e.message?.includes("Requested entity was not found") || e.status === 404) {
            setNeedsKey(true);
            setError("API Key issue. Please select a valid paid project key.");
        } else {
            setError(e.message || "Video generation failed.");
        }
    } finally {
        setLoading(false);
        setStatus('');
    }
  };

  const [error, setError] = useState<string | null>(null);

  return (
    <div className="flex flex-col h-full max-w-5xl mx-auto w-full p-6 gap-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-full">
            {/* Left: Controls */}
            <div className="space-y-6">
                <div className="space-y-2">
                    <h2 className="text-2xl font-bold text-white">Veo Video Generator</h2>
                    <p className="text-slate-400">Generate high-quality videos from text prompts using Veo 3.1.</p>
                </div>

                {needsKey && (
                    <div className="bg-amber-500/10 border border-amber-500/50 p-4 rounded-xl space-y-3">
                         <div className="flex items-center gap-2 text-amber-200 font-medium">
                            <AlertCircle className="w-5 h-5" />
                            Billing Account Required
                         </div>
                         <p className="text-sm text-amber-200/80">Veo models require a paid billing project. Please select a project.</p>
                         <button onClick={selectKey} className="text-sm bg-amber-600 hover:bg-amber-500 text-white px-4 py-2 rounded-lg font-medium transition-colors">
                            Select API Key
                         </button>
                         <div className="text-xs text-amber-200/60">
                            See <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" className="underline hover:text-white">billing docs</a> for details.
                         </div>
                    </div>
                )}

                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">Prompt</label>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="w-full bg-slate-800 border border-slate-700 rounded-xl p-4 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500 h-32 resize-none"
                            placeholder="Describe the video you want to generate..."
                        />
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                         <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">Resolution</label>
                            <select 
                                value={config.resolution}
                                onChange={(e) => setConfig({...config, resolution: e.target.value})}
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-purple-500 outline-none"
                            >
                                <option value="720p">720p (Faster)</option>
                                <option value="1080p">1080p (High Quality)</option>
                            </select>
                         </div>
                         <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">Aspect Ratio</label>
                            <select 
                                value={config.aspectRatio}
                                onChange={(e) => setConfig({...config, aspectRatio: e.target.value})}
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:ring-2 focus:ring-purple-500 outline-none"
                            >
                                <option value="16:9">16:9 (Landscape)</option>
                                <option value="9:16">9:16 (Portrait)</option>
                            </select>
                         </div>
                    </div>

                    <button
                        onClick={generateVideo}
                        disabled={loading || !prompt}
                        className="w-full bg-purple-600 hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed text-white py-4 rounded-xl font-semibold text-lg shadow-lg hover:shadow-purple-500/20 transition-all flex items-center justify-center gap-2"
                    >
                        {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Video className="w-5 h-5" />}
                        {loading ? 'Generating...' : 'Generate Video'}
                    </button>
                    
                    {status && <p className="text-center text-sm text-slate-400 animate-pulse">{status}</p>}
                    {error && <p className="text-center text-sm text-red-400">{error}</p>}
                </div>
            </div>

            {/* Right: Preview */}
            <div className="bg-black/40 rounded-2xl border border-slate-800 flex items-center justify-center relative overflow-hidden">
                {videoUri ? (
                    <div className="w-full h-full flex flex-col">
                        <video 
                            src={videoUri} 
                            controls 
                            autoPlay 
                            loop 
                            className="w-full h-full object-contain"
                        />
                    </div>
                ) : (
                    <div className="text-center p-8 space-y-4 opacity-30">
                        <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center mx-auto">
                            <Play className="w-8 h-8 text-slate-400" />
                        </div>
                        <p className="text-slate-400">Video preview will appear here</p>
                    </div>
                )}
            </div>
        </div>
    </div>
  );
};

// --- Multimodal Chat Component ---

const ChatView = () => {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'model', text: 'Hello! I am a multimodal assistant. You can ask me questions or show me images.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setSelectedImage(e.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  const sendMessage = async () => {
    if ((!input.trim() && !selectedImage) || isLoading) return;

    const userMsg: Message = { role: 'user', text: input, image: selectedImage || undefined };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setSelectedImage(null);
    setIsLoading(true);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const model = ai.getGenerativeModel({ model: 'gemini-2.5-flash' });
      
      let promptParts: any[] = [];
      if (userMsg.image) {
        promptParts.push({
            inlineData: {
                data: userMsg.image.split(',')[1],
                mimeType: 'image/jpeg' // Assuming jpeg for simplicity, realistically detect
            }
        });
      }
      if (userMsg.text) {
        promptParts.push({ text: userMsg.text });
      }

      // Stream response
      // For simplicity in this demo, using generateContentStream directly instead of chat history state for API
      // In a production app, manage chat history properly with ai.chats.create
      const streamResult = await ai.models.generateContentStream({
        model: 'gemini-2.5-flash',
        contents: { parts: promptParts }
      });

      setMessages(prev => [...prev, { role: 'model', text: '' }]);

      for await (const chunk of streamResult) {
         const text = chunk.text();
         setMessages(prev => {
             const newMsgs = [...prev];
             const lastMsg = newMsgs[newMsgs.length - 1];
             if (lastMsg.role === 'model') {
                 lastMsg.text += text;
             }
             return newMsgs;
         });
      }

    } catch (e: any) {
        console.error(e);
        setMessages(prev => [...prev, { role: 'model', text: "Error: " + (e.message || "Failed to generate response.") }]);
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full border-x border-slate-800 bg-slate-900">
       <div className="flex-1 overflow-y-auto p-4 space-y-4">
           {messages.map((msg, idx) => (
               <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                   <div className={`max-w-[80%] rounded-2xl p-4 ${msg.role === 'user' ? 'bg-emerald-600 text-white' : 'bg-slate-800 text-slate-100'}`}>
                       {msg.image && (
                           <img src={msg.image} alt="User upload" className="max-w-full h-auto rounded-lg mb-2 max-h-60 object-cover" />
                       )}
                       <p className="whitespace-pre-wrap">{msg.text}</p>
                   </div>
               </div>
           ))}
           <div ref={messagesEndRef} />
       </div>
       
       <div className="p-4 border-t border-slate-800 bg-slate-900/95 backdrop-blur">
           {selectedImage && (
               <div className="mb-2 inline-flex items-center gap-2 bg-slate-800 px-3 py-1 rounded-full text-sm">
                   <span className="text-slate-300">Image attached</span>
                   <button onClick={() => setSelectedImage(null)} className="text-slate-500 hover:text-white">&times;</button>
               </div>
           )}
           <div className="flex items-center gap-3">
               <label className="p-3 rounded-full hover:bg-slate-800 cursor-pointer text-slate-400 hover:text-emerald-500 transition-colors">
                   <input type="file" accept="image/*" className="hidden" onChange={handleImageSelect} />
                   <ImageIcon className="w-6 h-6" />
               </label>
               <input
                   type="text"
                   value={input}
                   onChange={(e) => setInput(e.target.value)}
                   onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
                   placeholder="Type a message..."
                   className="flex-1 bg-slate-800 text-white rounded-full px-5 py-3 focus:outline-none focus:ring-2 focus:ring-emerald-500"
               />
               <button 
                onClick={sendMessage}
                disabled={isLoading || (!input && !selectedImage)}
                className="p-3 bg-emerald-600 rounded-full text-white hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
               >
                   {isLoading ? <Loader2 className="w-6 h-6 animate-spin" /> : <Send className="w-6 h-6" />}
               </button>
           </div>
       </div>
    </div>
  );
};

// --- Main App Component ---

const App = () => {
  const [activeTab, setActiveTab] = useState<Tab>('live');

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-50">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="flex-1 overflow-hidden relative">
        <div className="absolute inset-0 overflow-y-auto">
          {activeTab === 'live' && <LiveView />}
          {activeTab === 'veo' && <VeoView />}
          {activeTab === 'chat' && <ChatView />}
        </div>
      </main>
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
