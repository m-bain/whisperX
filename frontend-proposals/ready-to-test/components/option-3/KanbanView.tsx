'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import { Transcription } from '../../types';
import { formatDuration, formatFileSize, formatDate } from '../../utils/formatters';
import { ExportDropdown } from '../option-1/ExportDropdown';

interface KanbanViewProps {
  transcriptions: Transcription[];
  isLoading: boolean;
  onDelete: (id: string) => void;
}

type Status = 'queued' | 'processing' | 'completed' | 'failed';

const COLUMNS: { id: Status; title: string; icon: string; color: string }[] = [
  { id: 'queued', title: 'In Coda', icon: '⏱', color: 'blue' },
  { id: 'processing', title: 'In Elaborazione', icon: '⟳', color: 'violet' },
  { id: 'completed', title: 'Completati', icon: '✓', color: 'emerald' },
  { id: 'failed', title: 'Falliti', icon: '✗', color: 'red' },
];

export function KanbanView({ transcriptions, isLoading, onDelete }: KanbanViewProps) {
  const [draggedId, setDraggedId] = useState<string | null>(null);

  const getTranscriptionsByStatus = (status: Status) => {
    return transcriptions.filter((t) => t.status === status);
  };

  const handleDragStart = (e: React.DragEvent, id: string) => {
    setDraggedId(id);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragEnd = () => {
    setDraggedId(null);
  };

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {COLUMNS.map((column) => (
          <div key={column.id} className="space-y-3">
            <div className="h-12 bg-gray-800/30 rounded-lg animate-pulse" />
            <div className="h-32 bg-gray-800/30 rounded-lg animate-pulse" />
            <div className="h-32 bg-gray-800/30 rounded-lg animate-pulse" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {COLUMNS.map((column) => {
        const items = getTranscriptionsByStatus(column.id);

        return (
          <motion.div
            key={column.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col"
          >
            {/* Column Header */}
            <div className={`bg-${column.color}-500/10 border border-${column.color}-500/30 rounded-xl p-4 mb-4`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{column.icon}</span>
                  <h3 className={`font-semibold text-${column.color}-300`}>
                    {column.title}
                  </h3>
                </div>
                <span className={`px-2 py-1 bg-${column.color}-500/20 rounded-lg text-sm font-medium text-${column.color}-300`}>
                  {items.length}
                </span>
              </div>
            </div>

            {/* Column Content */}
            <div className="flex-1 space-y-3 min-h-[200px]">
              <AnimatePresence mode="popLayout">
                {items.length === 0 ? (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="h-32 border-2 border-dashed border-gray-700/50 rounded-xl flex items-center justify-center text-gray-600"
                  >
                    <p className="text-sm">Nessun elemento</p>
                  </motion.div>
                ) : (
                  items.map((transcription, index) => (
                    <motion.div
                      key={transcription.id}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      transition={{ delay: index * 0.05 }}
                      draggable
                      onDragStart={(e) => handleDragStart(e, transcription.id)}
                      onDragEnd={handleDragEnd}
                      whileHover={{ scale: 1.02 }}
                      whileDrag={{ scale: 1.05, rotate: 2 }}
                      className={`bg-gray-800/50 border border-gray-700/50 rounded-xl p-4 cursor-move ${
                        draggedId === transcription.id ? 'opacity-50' : ''
                      }`}
                    >
                      {/* Card Header */}
                      <div className="mb-3">
                        <h4 className="font-medium text-gray-200 text-sm mb-1 line-clamp-2">
                          {transcription.fileName}
                        </h4>
                        <p className="text-xs text-gray-500">
                          {formatFileSize(transcription.fileSize)}
                        </p>
                      </div>

                      {/* Card Details */}
                      <div className="space-y-2 mb-3">
                        {transcription.durationSeconds && (
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-gray-500">Durata:</span>
                            <span className="text-gray-300">
                              {formatDuration(transcription.durationSeconds)}
                            </span>
                          </div>
                        )}

                        {transcription.detectedLanguage && (
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-gray-500">Lingua:</span>
                            <span className="text-gray-300 uppercase">
                              {transcription.detectedLanguage}
                            </span>
                          </div>
                        )}

                        <div className="flex items-center justify-between text-xs">
                          <span className="text-gray-500">Data:</span>
                          <span className="text-gray-400">
                            {formatDate(transcription.createdAt)}
                          </span>
                        </div>
                      </div>

                      {/* Card Actions */}
                      <div className="flex items-center gap-2 pt-3 border-t border-gray-700/50">
                        {transcription.status === 'completed' && transcription.transcriptionText && (
                          <ExportDropdown transcription={transcription} />
                        )}

                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={(e) => {
                            e.stopPropagation();
                            if (confirm('Sei sicuro di voler eliminare questa trascrizione?')) {
                              onDelete(transcription.id);
                            }
                          }}
                          className="ml-auto p-2 hover:bg-red-500/10 rounded-lg text-red-400 transition-colors"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </motion.button>
                      </div>
                    </motion.div>
                  ))
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
