const API_BASE = 'http://localhost:8000';

export const api = {
    // Create a new session
    createSession: async () => {
        const res = await fetch(`${API_BASE}/session/create`, { method: 'POST' });
        if (!res.ok) throw new Error('Failed to create session');
        return res.json();
    },

    // Generate image from text
    generate: async (sessionId, { style_preset, scene_text, width = 1024, height = 1024 }) => {
        const res = await fetch(`${API_BASE}/session/${sessionId}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                style_preset,
                scene_text,
                width,
                height
            })
        });
        if (!res.ok) throw new Error('Generation failed');
        return res.json();
    },

    // Get image URL
    getImageUrl: (sessionId, version) => {
        return `${API_BASE}/session/${sessionId}/image/${version}`;
    }
};
