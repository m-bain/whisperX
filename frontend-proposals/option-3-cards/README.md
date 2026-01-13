# Option 3: Card-based Modern Dashboard

Dashboard moderna con cards e animazioni fluide ispirata a Notion e Framer.

## 🎨 Design Filosofia

"Beautiful & Functional" - Focus su UX moderna e piacevole. Perfetta per tutti gli utenti che vogliono un'interfaccia bella e facile da usare.

## ✨ Features Uniche

### 🎴 Card-based Layout

- **Grid di cards responsive**: Ogni trascrizione è una card con hover effects
- **Drag & drop reordering**: Riordina le tue trascrizioni
- **Quick actions**: Pulsanti veloci su ogni card
- **Preview on hover**: Anteprima testo al passaggio del mouse
- **Modal full-screen**: Dettagli completi in modal elegante

### 🎭 Animations & Micro-interactions

- **Framer Motion**: Animazioni fluide ovunque
- **Loading skeletons**: Placeholder eleganti durante caricamento
- **Page transitions**: Transizioni smooth tra viste
- **Success animations**: Celebra i completamenti
- **Drag & drop upload**: Con animazioni feedback

### 🎯 Tutte le funzionalità base, più:

- **Kanban view**: Vista a colonne (Queued | Processing | Completed)
- **Grid/List toggle**: Passa tra vista griglia e lista
- **Card customization**: Scegli info da mostrare su ogni card
- **Smart search**: Ricerca full-text con highlight
- **Keyboard shortcuts**: Navigazione rapida con tastiera
- **Undo/Redo**: Per operazioni delete

## 📁 Componenti Principali

### Dashboard.tsx - Layout flessibile

```tsx
'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TranscriptionGrid } from './TranscriptionGrid';
import { TranscriptionKanban } from './TranscriptionKanban';
import { UploadModal } from './UploadModal';
import { DetailModal } from './DetailModal';

type ViewMode = 'grid' | 'list' | 'kanban';

export function Dashboard() {
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [selectedTranscription, setSelectedTranscription] = useState(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900/20 to-gray-900">
      {/* Animated Header */}
      <motion.header
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="sticky top-0 z-50 backdrop-blur-xl bg-gray-900/80 border-b border-white/10"
      >
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo & Title */}
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
                <span className="text-xl">🎙️</span>
              </div>
              <div>
                <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-violet-200 to-fuchsia-200">
                  WhisperX
                </h1>
                <p className="text-sm text-gray-400">Trascrizioni AI</p>
              </div>
            </div>

            {/* View Mode Toggle */}
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 bg-gray-800/50 p-1.5 rounded-lg">
                {[
                  { mode: 'grid', icon: '⊞' },
                  { mode: 'list', icon: '☰' },
                  { mode: 'kanban', icon: '⚏' },
                ].map(({ mode, icon }) => (
                  <button
                    key={mode}
                    onClick={() => setViewMode(mode as ViewMode)}
                    className={`px-3 py-2 rounded-md text-sm transition-all ${
                      viewMode === mode
                        ? 'bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white'
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    {icon}
                  </button>
                ))}
              </div>

              {/* Upload Button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsUploadModalOpen(true)}
                className="px-4 py-2 bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-lg font-medium text-white shadow-lg shadow-violet-500/25"
              >
                + Nuova Trascrizione
              </motion.button>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <AnimatePresence mode="wait">
          {viewMode === 'grid' && <TranscriptionGrid />}
          {viewMode === 'list' && <TranscriptionList />}
          {viewMode === 'kanban' && <TranscriptionKanban />}
        </AnimatePresence>
      </main>

      {/* Upload Modal */}
      <AnimatePresence>
        {isUploadModalOpen && (
          <UploadModal onClose={() => setIsUploadModalOpen(false)} />
        )}
      </AnimatePresence>

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedTranscription && (
          <DetailModal
            transcription={selectedTranscription}
            onClose={() => setSelectedTranscription(null)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
```

### TranscriptionCard.tsx - Card component

```tsx
'use client';

import { motion } from 'framer-motion';
import { StatusBadge } from './StatusBadge';
import { formatDuration, formatDate } from '../utils/formatters';
import type { Transcription } from '../types';

interface TranscriptionCardProps {
  transcription: Transcription;
  onClick: () => void;
  index: number;
}

export function TranscriptionCard({
  transcription,
  onClick,
  index,
}: TranscriptionCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ y: -4 }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      className="group relative"
    >
      {/* Gradient Glow Effect */}
      <div className="absolute -inset-0.5 bg-gradient-to-r from-violet-500 to-fuchsia-500 rounded-2xl opacity-0 group-hover:opacity-20 blur transition-opacity" />

      {/* Card */}
      <div
        onClick={onClick}
        className="relative bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl p-6 cursor-pointer transition-all hover:border-violet-500/50"
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            {/* Icon */}
            <motion.div
              animate={{ rotate: isHovered ? 360 : 0 }}
              transition={{ duration: 0.6 }}
              className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 border border-violet-500/30 flex items-center justify-center flex-shrink-0"
            >
              <span className="text-xl">🎵</span>
            </motion.div>

            {/* Title */}
            <div className="min-w-0 flex-1">
              <h3 className="font-semibold text-gray-100 truncate">
                {transcription.fileName}
              </h3>
              <p className="text-sm text-gray-400 mt-0.5">
                {formatDate(transcription.createdAt)}
              </p>
            </div>
          </div>

          {/* Status */}
          <StatusBadge status={transcription.status} />
        </div>

        {/* Preview Text */}
        {transcription.transcriptText && (
          <div className="mb-4">
            <p className="text-sm text-gray-300 line-clamp-2">
              {transcription.transcriptText}
            </p>
          </div>
        )}

        {/* Meta Info */}
        <div className="flex items-center gap-4 text-xs text-gray-400">
          {transcription.durationSeconds && (
            <div className="flex items-center gap-1.5">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {formatDuration(transcription.durationSeconds)}
            </div>
          )}

          {transcription.language && (
            <div className="flex items-center gap-1.5">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
              </svg>
              {transcription.language.toUpperCase()}
            </div>
          )}

          {transcription.speakers && (
            <div className="flex items-center gap-1.5">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
              {transcription.speakers.count} speaker
            </div>
          )}
        </div>

        {/* Quick Actions (show on hover) */}
        <AnimatePresence>
          {isHovered && transcription.status === 'completed' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="mt-4 pt-4 border-t border-gray-700/50 flex items-center gap-2"
            >
              <button className="flex-1 px-3 py-2 bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/30 rounded-lg text-sm text-violet-300 transition-colors">
                📄 Esporta
              </button>
              <button className="flex-1 px-3 py-2 bg-gray-700/30 hover:bg-gray-700/50 border border-gray-600/30 rounded-lg text-sm text-gray-300 transition-colors">
                👁️ Visualizza
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
```

### TranscriptionKanban.tsx - Vista Kanban

```tsx
'use client';

import { motion } from 'framer-motion';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { TranscriptionCard } from './TranscriptionCard';

const columns = [
  { id: 'queued', title: 'In Coda', color: 'from-blue-500 to-cyan-500' },
  { id: 'processing', title: 'Elaborazione', color: 'from-violet-500 to-fuchsia-500' },
  { id: 'completed', title: 'Completate', color: 'from-emerald-500 to-teal-500' },
  { id: 'failed', title: 'Fallite', color: 'from-red-500 to-orange-500' },
];

export function TranscriptionKanban() {
  const { transcriptions } = useTranscriptions();

  const transcriptionsByStatus = columns.reduce((acc, col) => {
    acc[col.id] = transcriptions.filter(t => t.status === col.id);
    return acc;
  }, {} as Record<string, Transcription[]>);

  return (
    <DragDropContext onDragEnd={handleDragEnd}>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="grid grid-cols-4 gap-6"
      >
        {columns.map((column) => (
          <div key={column.id} className="flex flex-col">
            {/* Column Header */}
            <div className="mb-4">
              <div className={`h-1 w-full bg-gradient-to-r ${column.color} rounded-full mb-3`} />
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-100">{column.title}</h3>
                <span className="px-2 py-1 bg-gray-800 rounded-md text-xs text-gray-400">
                  {transcriptionsByStatus[column.id]?.length || 0}
                </span>
              </div>
            </div>

            {/* Droppable Column */}
            <Droppable droppableId={column.id}>
              {(provided, snapshot) => (
                <div
                  ref={provided.innerRef}
                  {...provided.droppableProps}
                  className={`flex-1 space-y-3 p-3 rounded-xl transition-colors ${
                    snapshot.isDraggingOver
                      ? 'bg-gray-800/50 border-2 border-dashed border-violet-500/50'
                      : 'bg-transparent'
                  }`}
                >
                  {transcriptionsByStatus[column.id]?.map((transcription, index) => (
                    <Draggable
                      key={transcription.id}
                      draggableId={transcription.id}
                      index={index}
                    >
                      {(provided, snapshot) => (
                        <div
                          ref={provided.innerRef}
                          {...provided.draggableProps}
                          {...provided.dragHandleProps}
                          className={snapshot.isDragging ? 'opacity-50' : ''}
                        >
                          <TranscriptionCard
                            transcription={transcription}
                            index={index}
                          />
                        </div>
                      )}
                    </Draggable>
                  ))}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </div>
        ))}
      </motion.div>
    </DragDropContext>
  );
}
```

## 🚀 Setup

Requisiti Option 1, più:

```bash
npm install framer-motion react-beautiful-dnd
```

## 🎨 Design Features

### Gradients & Colors

```css
/* Gradient Backgrounds */
--gradient-purple: linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%);
--gradient-blue: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
--gradient-green: linear-gradient(135deg, #10b981 0%, #34d399 100%);

/* Card Glow Effects */
.card-glow {
  box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
}
```

### Animations

- **Page enter**: Fade + slide up
- **Card hover**: Lift + glow
- **Status change**: Morphing badge
- **Upload**: Progress ring animation
- **Success**: Confetti celebration
- **Delete**: Swipe out animation

## 🎯 Pro & Contro

### ✅ Pro

- **Bellissima**: Visivamente stunning
- **Moderna**: Trend UI/UX 2024
- **Intuitiva**: Facile da usare
- **Engaging**: Animazioni piacevoli
- **Flessibile**: Più viste disponibili

### ❌ Contro

- **Performance**: Animazioni pesanti su mobile
- **Complessità**: Più codice da mantenere
- **Learning curve**: Feature avanzate (Kanban)

## 🎴 Layout Views

### Grid View
- Cards 3-4 per riga
- Hover effects
- Quick actions

### List View
- Compatta
- Più informazioni visibili
- Sorting veloce

### Kanban View
- Drag & drop
- Visualizza stati
- Perfect for workflows

## 🔗 Recommended For

- **Tutti gli utenti**: Interface universal
- **Creative professionals**: Che apprezzano il design
- **Team collaboration**: Kanban view per workflows
- **Mobile users**: Responsive e touch-friendly (con qualche ottimizzazione)

## 🎭 Bonus Features

- **Dark/Light mode toggle**
- **Custom themes**: Scegli il tuo gradient
- **Keyboard shortcuts**: Power user features
- **Undo/Redo**: Per delete operations
- **Smart search**: Con fuzzy matching
- **Recently viewed**: Accesso rapido
