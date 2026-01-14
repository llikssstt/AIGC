import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Sparkles, AlertCircle } from 'lucide-react';
import { InkButton } from '../components/InkButton';
import { PoetryInput } from '../components/PoetryInput';
import { api } from '../services/api';

export const CreationPage = ({ onBack }) => {
    const [poem, setPoem] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleCreate = async () => {
        if (!poem.trim()) return;
        setLoading(true);
        setError(null);
        try {
            // 1. Create session
            const { session_id } = await api.createSession();

            // 2. Generate (using 'shuimo' style by default)
            const data = await api.generate(session_id, {
                style_preset: "水墨",
                scene_text: poem,
                width: 1024,
                height: 1024
            });

            setResult({
                imageUrl: data.image_url, // Relative path from API response
                prompt: data.prompt_card?.final_prompt
            });
        } catch (err) {
            console.error(err);
            setError("创作过程中墨水干涸了，请重试...");
        } finally {
            setLoading(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="container"
            style={{ paddingTop: '2rem', paddingBottom: '4rem', minHeight: '100vh' }}
        >
            <header style={{ display: 'flex', alignItems: 'center', marginBottom: '3rem' }}>
                <button
                    onClick={onBack}
                    style={{
                        display: 'flex', alignItems: 'center', gap: '8px',
                        color: 'var(--c-ink-light)', padding: '8px',
                        fontSize: '1rem'
                    }}
                >
                    <ArrowLeft size={20} /> 返回
                </button>
                <span style={{ marginLeft: 'auto', fontStyle: 'italic', color: 'var(--c-seal)' }}>由于算力限制，请耐心等待</span>
            </header>

            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
                {!result ? (
                    <motion.div
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                    >
                        <h2 className="text-center" style={{ marginBottom: '2rem', fontSize: '2rem' }}>提笔挥毫</h2>

                        <div style={{ marginBottom: '2rem' }}>
                            <PoetryInput
                                value={poem}
                                onChange={setPoem}
                                disabled={loading}
                                placeholder="在此输入诗句，如：&#10;孤舟蓑笠翁，&#10;独钓寒江雪。"
                            />
                        </div>

                        {error && (
                            <div style={{ color: '#e74c3c', marginBottom: '1rem', textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                                <AlertCircle size={16} /> {error}
                            </div>
                        )}

                        <div className="flex-center">
                            <InkButton onClick={handleCreate} disabled={loading || !poem.trim()}>
                                {loading ? (
                                    <span className="flex-center" style={{ gap: '8px' }}>
                                        <motion.div
                                            animate={{ rotate: 360 }}
                                            transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                                        >
                                            <Sparkles size={18} />
                                        </motion.div>
                                        研磨中...
                                    </span>
                                ) : (
                                    <>生成意境</>
                                )}
                            </InkButton>
                        </div>
                    </motion.div>
                ) : (
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className="text-center"
                    >
                        <div style={{
                            background: '#fff',
                            padding: '16px',
                            borderRadius: '2px',
                            boxShadow: 'var(--shadow-elevation)',
                            display: 'inline-block',
                            marginBottom: '2rem'
                        }}>
                            <img
                                src={`http://localhost:8000${result.imageUrl}`}
                                alt="Generated Result"
                                style={{ maxWidth: '100%', maxHeight: '70vh', display: 'block' }}
                            />
                        </div>

                        <div className="text-center">
                            <p style={{ fontSize: '1.5rem', marginBottom: '1rem', fontFamily: 'var(--font-serif)' }}>{poem}</p>
                            <div className="flex-center" style={{ gap: '1rem' }}>
                                <InkButton onClick={() => setResult(null)} variant="secondary">再作一首</InkButton>
                            </div>
                        </div>
                    </motion.div>
                )}
            </div>
        </motion.div>
    );
};
