import React, { useRef, useState, useEffect } from 'react';
import styles from './MaskCanvas.module.css';

export const MaskCanvas = ({ imageUrl, onMaskChange, width = 512, height = 512 }) => {
    const canvasRef = useRef(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [brushSize, setBrushSize] = useState(30);
    const [imageLoaded, setImageLoaded] = useState(false);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Load background image
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            // Draw image as background (we'll overlay the mask)
            ctx.drawImage(img, 0, 0, width, height);
            setImageLoaded(true);
        };
        img.src = imageUrl;
    }, [imageUrl, width, height]);

    const getPos = (e) => {
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        if (e.touches) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    };

    const startDrawing = (e) => {
        setIsDrawing(true);
        draw(e);
    };

    const stopDrawing = () => {
        setIsDrawing(false);
        exportMask();
    };

    const draw = (e) => {
        if (!isDrawing) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const pos = getPos(e);

        // Draw white circle (mask area)
        ctx.fillStyle = 'rgba(255, 0, 0, 0.5)'; // Red overlay for visibility
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
        ctx.fill();
    };

    const exportMask = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Create a new canvas for the mask (black/white only)
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = width;
        maskCanvas.height = height;
        const maskCtx = maskCanvas.getContext('2d');

        // Get image data from main canvas
        const imageData = ctx.getImageData(0, 0, width, height);
        const data = imageData.data;

        // Create mask: where we drew (red areas) becomes white, rest is black
        const maskData = maskCtx.createImageData(width, height);
        for (let i = 0; i < data.length; i += 4) {
            // Check if this pixel has red overlay (R > 200, G < 100)
            const isMarked = data[i] > 200 && data[i + 1] < 150;
            maskData.data[i] = isMarked ? 255 : 0;     // R
            maskData.data[i + 1] = isMarked ? 255 : 0; // G
            maskData.data[i + 2] = isMarked ? 255 : 0; // B
            maskData.data[i + 3] = 255;                // A
        }
        maskCtx.putImageData(maskData, 0, 0);

        // Export as blob
        maskCanvas.toBlob((blob) => {
            if (onMaskChange) {
                onMaskChange(blob);
            }
        }, 'image/png');
    };

    const clearMask = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Reload original image
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            ctx.clearRect(0, 0, width, height);
            ctx.drawImage(img, 0, 0, width, height);
        };
        img.src = imageUrl;
    };

    return (
        <div className={styles.container}>
            <canvas
                ref={canvasRef}
                width={width}
                height={height}
                className={styles.canvas}
                onMouseDown={startDrawing}
                onMouseUp={stopDrawing}
                onMouseMove={draw}
                onMouseLeave={stopDrawing}
                onTouchStart={startDrawing}
                onTouchEnd={stopDrawing}
                onTouchMove={draw}
            />
            <div className={styles.controls}>
                <label className={styles.label}>
                    笔刷大小: {brushSize}
                    <input
                        type="range"
                        min="10"
                        max="100"
                        value={brushSize}
                        onChange={(e) => setBrushSize(parseInt(e.target.value))}
                        className={styles.slider}
                    />
                </label>
                <button onClick={clearMask} className={styles.clearBtn}>
                    清除蒙版
                </button>
            </div>
            <p className={styles.hint}>在需要修改的区域涂抹红色蒙版</p>
        </div>
    );
};
