'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { Transcription } from '../../types';
import { TranscriptionCard } from './TranscriptionCard';

interface GridViewProps {
  transcriptions: Transcription[];
  isLoading: boolean;
  onDelete: (id: string) => void;
}

export function GridView({ transcriptions, isLoading, onDelete }: GridViewProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="h-64 bg-gray-800/30 rounded-xl border border-gray-700/50 animate-pulse"
          />
        ))}
      </div>
    );
  }

  if (transcriptions.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-12"
      >
        <div className="w-20 h-20 mx-auto mb-4 bg-gray-800/50 rounded-full flex items-center justify-center">
          <span className="text-4xl">📁</span>
        </div>
        <p className="text-gray-400 text-lg mb-2">Nessuna trascrizione</p>
        <p className="text-gray-500 text-sm">Carica un file audio per iniziare</p>
      </motion.div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      <AnimatePresence mode="popLayout">
        {transcriptions.map((transcription, index) => (
          <TranscriptionCard
            key={transcription.id}
            transcription={transcription}
            onDelete={onDelete}
            index={index}
          />
        ))}
      </AnimatePresence>
    </div>
  );
}
