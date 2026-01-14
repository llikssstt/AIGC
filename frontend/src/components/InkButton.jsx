import React from 'react';
import { motion } from 'framer-motion';
import styles from './InkButton.module.css';

export const InkButton = ({ children, onClick, variant = 'primary', className = '' }) => {
    return (
        <motion.button
            className={`${styles.button} ${styles[variant]} ${className}`}
            onClick={onClick}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
        >
            <span className={styles.ink} />
            <span className={styles.content}>{children}</span>
        </motion.button>
    );
};
