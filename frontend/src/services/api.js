const API_BASE = 'http://localhost:8000';

export const api = {
    // Create a new session
    createSession: async () => {
        const res = await fetch(`${API_BASE}/session/create`, { method: 'POST' });
        if (!res.ok) throw new Error('Failed to create session');
        return res.json();
    },

    // List all sessions for gallery
    listSessions: async () => {
        const res = await fetch(`${API_BASE}/sessions`);
        if (!res.ok) throw new Error('Failed to list sessions');
        return res.json();
    },

    // Generate image from text
    generate: async (sessionId, { style_preset, scene_text, seed = -1, steps = 30, cfg = 7.5, width = 1024, height = 1024 }) => {
        const res = await fetch(`${API_BASE}/session/${sessionId}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                style_preset,
                scene_text,
                seed,
                steps,
                cfg,
                width,
                height
            })
        });
        if (!res.ok) throw new Error('Generation failed');
        return res.json();
    },

    // Edit image with mask
    edit: async (sessionId, {
        maskBlob,
        edit_text,
        seed = -1,
        steps = 30,
        cfg = 7.5,
        strength = 0.6,
        grow_pixels = 8,
        blur_sigma = 12.0,
        invert_mask = false
    }) => {
        const formData = new FormData();
        formData.append('mask', maskBlob, 'mask.png');
        formData.append('edit_text', edit_text);
        formData.append('seed', seed.toString());
        formData.append('steps', steps.toString());
        formData.append('cfg', cfg.toString());
        formData.append('strength', strength.toString());
        formData.append('grow_pixels', grow_pixels.toString());
        formData.append('blur_sigma', blur_sigma.toString());
        formData.append('invert_mask', invert_mask.toString());

        const res = await fetch(`${API_BASE}/session/${sessionId}/edit`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) throw new Error('Edit failed');
        return res.json();
    },

    // Get session history
    getHistory: async (sessionId) => {
        const res = await fetch(`${API_BASE}/session/${sessionId}/history`);
        if (!res.ok) throw new Error('Failed to fetch history');
        return res.json();
    },

    // Revert to a previous version
    revert: async (sessionId, version) => {
        const formData = new FormData();
        formData.append('version', version.toString());
        const res = await fetch(`${API_BASE}/session/${sessionId}/revert`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) throw new Error('Revert failed');
        return res.json();
    },

    // Get image URL
    getImageUrl: (sessionId, version) => {
        return `${API_BASE}/session/${sessionId}/image/${version}`;
    },

    // Get thumbnail URL
    getThumbnailUrl: (sessionId, version) => {
        return `${API_BASE}/session/${sessionId}/thumbnail/${version}`;
    }
};
