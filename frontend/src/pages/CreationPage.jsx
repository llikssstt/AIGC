import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Sparkles, AlertCircle, Edit3, Settings, ChevronDown, ChevronUp } from 'lucide-react';
import { InkButton } from '../components/InkButton';
import { PoetryInput } from '../components/PoetryInput';
import { api } from '../services/api';
import styles from './CreationPage.module.css';

const STYLE_OPTIONS = [
    { value: '水墨', label: '水墨 (Ink Wash)' },
    { value: '工笔', label: '工笔 (Gongbi)' },
    { value: '青绿', label: '青绿 (Qinglv)' },
];

export const CreationPage = ({ onBack, onEdit }) => {
    const [poem, setPoem] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [sessionId, setSessionId] = useState(null);

    // Style & Parameters
    const [style, setStyle] = useState('水墨');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [params, setParams] = useState({
        seed: -1,
        steps: 30,
        cfg: 7.5,
        width: 1024,
        height: 1024,
    });

    const updateParam = (key, value) => {
        setParams(prev => ({ ...prev, [key]: value }));
    };

    const handleCreate = async () => {
        if (!poem.trim()) return;
        setLoading(true);
        setError(null);
        try {
            const { session_id } = await api.createSession();
            setSessionId(session_id);

            const data = await api.generate(session_id, {
                style_preset: style,
                scene_text: poem,
                seed: params.seed,
                steps: params.steps,
                cfg: params.cfg,
                width: params.width,
                height: params.height,
            });

            setResult({
                version: data.version,
                imageUrl: api.getImageUrl(session_id, data.version),
                prompt: data.prompt_card?.final_prompt,
                actualSeed: data.prompt_card?.seed,
            });
        } catch (err) {
            console.error(err);
            setError("创作过程中墨水干涸了，请重试...");
        } finally {
            setLoading(false);
        }
    };

    const handleGoToEdit = () => {
        if (sessionId && result) {
            onEdit({
                sessionId,
                version: result.version,
                imageUrl: result.imageUrl
            });
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
            <header className={styles.header}>
                <button onClick={onBack} className={styles.backBtn}>
                    <ArrowLeft size={20} /> 返回
                </button>
                <span className={styles.notice}>由于算力限制，请耐心等待</span>
            </header>

            <div className={styles.main}>
                {!result ? (
                    <motion.div
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                    >
                        <h2 className="text-center" style={{ marginBottom: '2rem', fontSize: '2rem' }}>提笔挥毫</h2>

                        {/* Style Selector */}
                        <div className={styles.formGroup}>
                            <label className={styles.label}>Style 风格</label>
                            <select
                                value={style}
                                onChange={(e) => setStyle(e.target.value)}
                                className={styles.select}
                                disabled={loading}
                            >
                                {STYLE_OPTIONS.map(opt => (
                                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                                ))}
                            </select>
                        </div>

                        {/* Poetry Input */}
                        <div className={styles.formGroup}>
                            <label className={styles.label}>Scene / Poem 场景 / 诗歌</label>
                            <PoetryInput
                                value={poem}
                                onChange={setPoem}
                                disabled={loading}
                                placeholder="输入一句话或一首诗（支持多行）。系统会自动抽取意象并增强提示词。"
                            />
                        </div>

                        {/* Advanced Parameters Toggle */}
                        <button
                            onClick={() => setShowAdvanced(!showAdvanced)}
                            className={styles.advancedToggle}
                        >
                            <Settings size={16} />
                            高级参数
                            {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </button>

                        {/* Advanced Parameters Panel */}
                        <AnimatePresence>
                            {showAdvanced && (
                                <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    className={styles.advancedPanel}
                                >
                                    {/* Seed */}
                                    <div className={styles.paramRow}>
                                        <label>Seed 种子 (-1 随机)</label>
                                        <input
                                            type="number"
                                            value={params.seed}
                                            onChange={(e) => updateParam('seed', parseInt(e.target.value) || -1)}
                                            className={styles.numberInput}
                                        />
                                    </div>

                                    {/* Steps */}
                                    <div className={styles.paramRow}>
                                        <label>Steps 步骤: {params.steps}</label>
                                        <input
                                            type="range"
                                            min="10"
                                            max="80"
                                            value={params.steps}
                                            onChange={(e) => updateParam('steps', parseInt(e.target.value))}
                                            className={styles.slider}
                                        />
                                    </div>

                                    {/* CFG */}
                                    <div className={styles.paramRow}>
                                        <label>CFG: {params.cfg}</label>
                                        <input
                                            type="range"
                                            min="1"
                                            max="20"
                                            step="0.5"
                                            value={params.cfg}
                                            onChange={(e) => updateParam('cfg', parseFloat(e.target.value))}
                                            className={styles.slider}
                                        />
                                    </div>

                                    {/* Width */}
                                    <div className={styles.paramRow}>
                                        <label>Width 宽度: {params.width}</label>
                                        <input
                                            type="range"
                                            min="512"
                                            max="1536"
                                            step="64"
                                            value={params.width}
                                            onChange={(e) => updateParam('width', parseInt(e.target.value))}
                                            className={styles.slider}
                                        />
                                    </div>

                                    {/* Height */}
                                    <div className={styles.paramRow}>
                                        <label>Height 高度: {params.height}</label>
                                        <input
                                            type="range"
                                            min="512"
                                            max="1536"
                                            step="64"
                                            value={params.height}
                                            onChange={(e) => updateParam('height', parseInt(e.target.value))}
                                            className={styles.slider}
                                        />
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {error && (
                            <div className={styles.error}>
                                <AlertCircle size={16} /> {error}
                            </div>
                        )}

                        <div className="flex-center" style={{ marginTop: '2rem' }}>
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
                        <div className={styles.resultFrame}>
                            <img
                                src={result.imageUrl}
                                alt="Generated Result"
                                className={styles.resultImage}
                            />
                        </div>

                        <div className="text-center">
                            <p className={styles.poemDisplay}>{poem}</p>
                            {result.actualSeed && (
                                <p className={styles.seedInfo}>Seed: {result.actualSeed}</p>
                            )}
                            <div className="flex-center" style={{ gap: '1rem' }}>
                                <InkButton onClick={() => setResult(null)} variant="secondary">再作一首</InkButton>
                                <InkButton onClick={handleGoToEdit}>
                                    <Edit3 size={18} style={{ marginRight: '8px' }} />
                                    编辑此图
                                </InkButton>
                            </div>
                        </div>
                    </motion.div>
                )}
            </div>
        </motion.div>
    );
};
