import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { PenTool, Image as ImageIcon } from 'lucide-react'
import { InkButton } from './components/InkButton'
import { CreationPage } from './pages/CreationPage'
import { EditPage } from './pages/EditPage'
import { GalleryPage } from './pages/GalleryPage'
import './index.css'

function App() {
  const [view, setView] = useState('home'); // home | create | edit | gallery
  const [editContext, setEditContext] = useState(null); // { sessionId, version, imageUrl }

  const handleGoToEdit = (context) => {
    setEditContext(context);
    setView('edit');
  };

  const handleBackFromEdit = () => {
    setEditContext(null);
    setView('home');
  };

  if (view === 'create') {
    return (
      <CreationPage
        onBack={() => setView('home')}
        onEdit={handleGoToEdit}
      />
    );
  }

  if (view === 'edit' && editContext) {
    return (
      <EditPage
        sessionId={editContext.sessionId}
        initialVersion={editContext.version}
        initialImageUrl={editContext.imageUrl}
        onBack={handleBackFromEdit}
      />
    );
  }

  if (view === 'gallery') {
    return (
      <GalleryPage
        onBack={() => setView('home')}
        onEdit={handleGoToEdit}
      />
    );
  }

  return (
    <div className="container" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -30 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="text-center"
        >
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 'var(--sp-lg)' }}>
            <div style={{
              width: '80px', height: '80px',
              borderRadius: '50%',
              backgroundColor: 'var(--c-seal)',
              color: 'white',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: 'var(--shadow-elevation)'
            }}>
              <PenTool size={40} />
            </div>
          </div>

          <h1 style={{ fontSize: '3rem', marginBottom: 'var(--sp-sm)', letterSpacing: '0.2em' }}>古诗·绘意</h1>
          <p className="text-faint" style={{ fontSize: '1.25rem', marginBottom: 'var(--sp-xl)' }}>Digital Ink & Poetry</p>

          <div style={{ display: 'flex', gap: 'var(--sp-md)', justifyContent: 'center' }}>
            <InkButton onClick={() => setView('create')}>
              开始创作
            </InkButton>
            <InkButton variant="secondary" onClick={() => setView('gallery')}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <ImageIcon size={18} /> 画廊
              </span>
            </InkButton>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

export default App
