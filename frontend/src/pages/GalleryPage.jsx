import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ArrowLeft, Image as ImageIcon, Trash2, Edit3 } from 'lucide-react';
import { api } from '../services/api';
import styles from './GalleryPage.module.css';

export const GalleryPage = ({ onBack, onEdit }) => {
    const [sessions, setSessions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedSession, setSelectedSession] = useState(null);

    useEffect(() => {
        loadSessions();
    }, []);

    const loadSessions = async () => {
        try {
            const data = await api.listSessions();
            setSessions(data);
        } catch (err) {
            console.error('Failed to load sessions:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleSessionClick = (session) => {
        setSelectedSession(session);
    };

    const handleEdit = () => {
        if (selectedSession) {
            onEdit({
                sessionId: selectedSession.session_id,
                version: selectedSession.current_version,
                imageUrl: api.getImageUrl(selectedSession.session_id, selectedSession.current_version),
            });
        }
    };

    const formatDate = (isoString) => {
        const date = new Date(isoString);
        return date.toLocaleDateString('zh-CN', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
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
                <h1 className={styles.title}>
                    <ImageIcon size={24} /> 画廊
                </h1>
            </header>

            {loading ? (
                <div className={styles.loading}>墨迹渐现...</div>
            ) : sessions.length === 0 ? (
                <div className={styles.empty}>
                    <p>画廊空空如也</p>
                    <p className={styles.emptyHint}>开始创作，留下你的第一幅作品</p>
                </div>
            ) : (
                <div className={styles.grid}>
                    {sessions.map((session, index) => (
                        <motion.div
                            key={session.session_id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.05 }}
                            className={`${styles.card} ${selectedSession?.session_id === session.session_id ? styles.selected : ''}`}
                            onClick={() => handleSessionClick(session)}
                        >
                            <div className={styles.imageWrapper}>
                                <img
                                    src={`http://localhost:8000${session.thumbnail_url}`}
                                    alt={`Session ${session.session_id}`}
                                    className={styles.thumbnail}
                                />
                                {session.total_versions > 1 && (
                                    <span className={styles.versionBadge}>
                                        {session.total_versions} 版本
                                    </span>
                                )}
                            </div>
                            <div className={styles.cardInfo}>
                                <span className={styles.styleBadge}>{session.style_preset || '未知'}</span>
                                <span className={styles.date}>{formatDate(session.updated_at)}</span>
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}

            {/* Detail Panel */}
            {selectedSession && (
                <motion.div
                    initial={{ x: 300, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    className={styles.detailPanel}
                >
                    <h3>作品详情</h3>
                    <img
                        src={api.getImageUrl(selectedSession.session_id, selectedSession.current_version)}
                        alt="Selected"
                        className={styles.previewImage}
                    />
                    <div className={styles.detailInfo}>
                        <p><strong>风格:</strong> {selectedSession.style_preset || '未知'}</p>
                        <p><strong>版本数:</strong> {selectedSession.total_versions}</p>
                        <p><strong>创建于:</strong> {formatDate(selectedSession.created_at)}</p>
                        <p><strong>更新于:</strong> {formatDate(selectedSession.updated_at)}</p>
                    </div>
                    <div className={styles.detailActions}>
                        <button onClick={handleEdit} className={styles.editBtn}>
                            <Edit3 size={16} /> 继续编辑
                        </button>
                        <button onClick={() => setSelectedSession(null)} className={styles.closeBtn}>
                            关闭
                        </button>
                    </div>
                </motion.div>
            )}
        </motion.div>
    );
};
