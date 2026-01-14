import React from 'react';
import styles from './PoetryInput.module.css';

export const PoetryInput = ({ value, onChange, placeholder = "在此挥毫...", disabled = false }) => {
    return (
        <div className={styles.wrapper}>
            <textarea
                className={styles.input}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                disabled={disabled}
                rows={4}
            />
            <div className={styles.cornerT} />
            <div className={styles.cornerB} />
        </div>
    );
};
