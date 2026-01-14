import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, Wand2, History, RotateCcw, AlertCircle, Settings, ChevronDown, ChevronUp } from 'lucide-react';
import { InkButton } from '../components/InkButton';
import { MaskCanvas } from '../components/MaskCanvas';
import { api } from '../services/api';
import styles from './EditPage.module.css';

export const EditPage = ({ sessionId, initialVersion, initialImageUrl, onBack }) => {
    const [currentVersion, setCurrentVersion] = useState(initialVersion);
    const [imageUrl, setImageUrl] = useState(initialImageUrl);
    const [editText, setEditText] = useState('');
    const [maskBlob, setMaskBlob] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [history, setHistory] = useState([]);
    const [showHistory, setShowHistory] = useState(false);

    // Edit parameters
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [params, setParams] = useState({
        seed: -1,
        steps: 30,
        cfg: 7.5,
        strength: 0.6,
        grow_pixels: 8,
        blur_sigma: 12.0,
        invert_mask: false,
    });

    const updateParam = (key, value) => {
        setParams(prev => ({ ...prev, [key]: value }));
    };

    // Load history on mount
    useEffect(() => {
        loadHistory();
    }, [sessionId]);

    const loadHistory = async () => {
        try {
            const items = await api.getHistory(sessionId);
            setHistory(items);
        } catch (err) {
            console.error('Failed to load history:', err);
        }
    };

    const handleEdit = async () => {
        if (!maskBlob || !editText.trim()) {
            setError('请先涂抹蒙版并输入修改指令');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const result = await api.edit(sessionId, {
                maskBlob,
                edit_text: editText,
                seed: params.seed,
                steps: params.steps,
                cfg: params.cfg,
                strength: params.strength,
                grow_pixels: params.grow_pixels,
                blur_sigma: params.blur_sigma,
                invert_mask: params.invert_mask,
            });

            setCurrentVersion(result.version);
            setImageUrl(api.getImageUrl(sessionId, result.version));
            setEditText('');
            setMaskBlob(null);
            loadHistory();
        } catch (err) {
            console.error(err);
            setError('编辑失败，请重试');
        } finally {
            setLoading(false);
        }
    };

    const handleRevert = async (version) => {
        try {
            await api.revert(sessionId, version);
            setCurrentVersion(version);
            setImageUrl(api.getImageUrl(sessionId, version));
            loadHistory();
            setShowHistory(false);
        } catch (err) {
            console.error(err);
            setError('回退失败');
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className={styles.container}
        >
            <header className={styles.header}>
                <button onClick={onBack} className={styles.backBtn}>
                    <ArrowLeft size={20} /> 返回
                </button>
                <span className={styles.versionBadge}>v{currentVersion}</span>
                <button
                    onClick={() => setShowHistory(!showHistory)}
                    className={styles.historyBtn}
                >
                    <History size={18} /> 历史
                </button>
            </header>

            <div className={styles.main}>
                <div className={styles.canvasSection}>
                    <h2 className={styles.sectionTitle}>涂抹需要修改的区域</h2>
                    <MaskCanvas
                        imageUrl={imageUrl}
                        onMaskChange={setMaskBlob}
                        width={512}
                        height={512}
                    />
                </div>

                <div className={styles.controlSection}>
                    <h2 className={styles.sectionTitle}>修改指令</h2>
                    <textarea
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        placeholder="例如：把衣服换成红色 / 添加一只小船 / 删除背景中的树"
                        className={styles.textarea}
                        rows={3}
                        disabled={loading}
                    />

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

                                {/* Strength */}
                                <div className={styles.paramRow}>
                                    <label>Strength 强度: {params.strength}</label>
                                    <input
                                        type="range"
                                        min="0.1"
                                        max="1.0"
                                        step="0.05"
                                        value={params.strength}
                                        onChange={(e) => updateParam('strength', parseFloat(e.target.value))}
                                        className={styles.slider}
                                    />
                                    <span className={styles.paramHint}>0.3-0.5 微调 | 0.7-0.9 大改</span>
                                </div>

                                {/* Grow Pixels */}
                                <div className={styles.paramRow}>
                                    <label>Grow Pixels 膨胀: {params.grow_pixels}px</label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="32"
                                        value={params.grow_pixels}
                                        onChange={(e) => updateParam('grow_pixels', parseInt(e.target.value))}
                                        className={styles.slider}
                                    />
                                </div>

                                {/* Blur Sigma */}
                                <div className={styles.paramRow}>
                                    <label>Blur 羽化: {params.blur_sigma}</label>
                                    <input
                                        type="range"
                                        min="0"
                                        max="30"
                                        step="1"
                                        value={params.blur_sigma}
                                        onChange={(e) => updateParam('blur_sigma', parseFloat(e.target.value))}
                                        className={styles.slider}
                                    />
                                </div>

                                {/* Invert Mask */}
                                <div className={styles.paramRow}>
                                    <label className={styles.checkboxLabel}>
                                        <input
                                            type="checkbox"
                                            checked={params.invert_mask}
                                            onChange={(e) => updateParam('invert_mask', e.target.checked)}
                                        />
                                        反转蒙版 (修改未涂抹区域)
                                    </label>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {error && (
                        <div className={styles.error}>
                            <AlertCircle size={16} /> {error}
                        </div>
                    )}

                    <InkButton onClick={handleEdit} disabled={loading || !maskBlob}>
                        {loading ? (
                            <span className={styles.loadingText}>
                                <motion.span
                                    animate={{ rotate: 360 }}
                                    transition={{ repeat: Infinity, duration: 1.5, ease: 'linear' }}
                                    style={{ display: 'inline-block' }}
                                >
                                    <Wand2 size={18} />
                                </motion.span>
                                修改中...
                            </span>
                        ) : (
                            <>
                                <Wand2 size={18} style={{ marginRight: '8px' }} />
                                应用修改
                            </>
                        )}
                    </InkButton>
                </div>
            </div>

            {/* History Panel */}
            {showHistory && (
                <motion.div
                    initial={{ x: 300, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    className={styles.historyPanel}
                >
                    <h3>编辑历史</h3>
                    <div className={styles.historyList}>
                        {history.map((item) => (
                            <div
                                key={item.version}
                                className={`${styles.historyItem} ${item.version === currentVersion ? styles.active : ''}`}
                                onClick={() => handleRevert(item.version)}
                            >
                                <img
                                    src={api.getThumbnailUrl(sessionId, item.version)}
                                    alt={`v${item.version}`}
                                    className={styles.thumbnail}
                                />
                                <div className={styles.historyInfo}>
                                    <span className={styles.historyVersion}>v{item.version}</span>
                                    <span className={styles.historyType}>{item.edit_type}</span>
                                    {item.edit_text && (
                                        <span className={styles.historyText}>{item.edit_text.slice(0, 20)}...</span>
                                    )}
                                </div>
                                {item.version !== currentVersion && (
                                    <RotateCcw size={16} className={styles.revertIcon} />
                                )}
                            </div>
                        ))}
                    </div>
                </motion.div>
            )}
        </motion.div>
    );
};
