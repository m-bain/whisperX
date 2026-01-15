'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import { Transcription } from '../../types';
import { formatDuration, formatFileSize, formatDate } from '../../utils/formatters';
import { StatusBadge } from '../option-1/StatusBadge';
import { ExportDropdown } from '../option-1/ExportDropdown';

interface ListViewProps {
  transcriptions: Transcription[];
  isLoading: boolean;
  onDelete: (id: string) => void;
}

export function ListView({ transcriptions, isLoading, onDelete }: ListViewProps) {
  const [sortBy, setSortBy] = useState<'date' | 'name' | 'duration'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const sortedTranscriptions = [...transcriptions].sort((a, b) => {
    let comparison = 0;

    switch (sortBy) {
      case 'date':
        comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
        break;
      case 'name':
        comparison = a.fileName.localeCompare(b.fileName);
        break;
      case 'duration':
        comparison = (a.durationSeconds || 0) - (b.durationSeconds || 0);
        break;
    }

    return sortOrder === 'asc' ? comparison : -comparison;
  });

  const toggleSort = (field: 'date' | 'name' | 'duration') => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const SortIcon = ({ active, order }: { active: boolean; order: 'asc' | 'desc' }) => (
    <svg
      className={`w-4 h-4 transition-all ${active ? 'text-emerald-400' : 'text-gray-600'}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      {order === 'asc' ? (
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
      ) : (
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
      )}
    </svg>
  );

  if (isLoading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-20 bg-gray-800/30 rounded-lg animate-pulse" />
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
          <span className="text-4xl">📋</span>
        </div>
        <p className="text-gray-400 text-lg mb-2">Nessuna trascrizione</p>
        <p className="text-gray-500 text-sm">Carica un file audio per iniziare</p>
      </motion.div>
    );
  }

  return (
    <div className="bg-gray-900/30 border border-gray-800/50 rounded-xl overflow-hidden">
      {/* Table Header */}
      <div className="grid grid-cols-12 gap-4 px-6 py-4 bg-gray-800/30 border-b border-gray-700/50 text-sm font-medium text-gray-400">
        <button
          onClick={() => toggleSort('name')}
          className="col-span-4 flex items-center gap-2 hover:text-emerald-400 transition-colors text-left"
        >
          File
          <SortIcon active={sortBy === 'name'} order={sortOrder} />
        </button>
        <div className="col-span-2">Stato</div>
        <button
          onClick={() => toggleSort('duration')}
          className="col-span-2 flex items-center gap-2 hover:text-emerald-400 transition-colors"
        >
          Durata
          <SortIcon active={sortBy === 'duration'} order={sortOrder} />
        </button>
        <button
          onClick={() => toggleSort('date')}
          className="col-span-2 flex items-center gap-2 hover:text-emerald-400 transition-colors"
        >
          Data
          <SortIcon active={sortBy === 'date'} order={sortOrder} />
        </button>
        <div className="col-span-2 text-right">Azioni</div>
      </div>

      {/* Table Body */}
      <div className="divide-y divide-gray-700/50">
        <AnimatePresence mode="popLayout">
          {sortedTranscriptions.map((transcription, index) => (
            <motion.div
              key={transcription.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ delay: index * 0.03 }}
              whileHover={{ backgroundColor: 'rgba(16, 185, 129, 0.05)' }}
              className="grid grid-cols-12 gap-4 px-6 py-4 items-center transition-colors"
            >
              {/* File Name */}
              <div className="col-span-4 min-w-0">
                <p className="font-medium text-gray-200 truncate mb-1">
                  {transcription.fileName}
                </p>
                <p className="text-xs text-gray-500">
                  {formatFileSize(transcription.fileSize)}
                  {transcription.detectedLanguage && ` • ${transcription.detectedLanguage.toUpperCase()}`}
                </p>
              </div>

              {/* Status */}
              <div className="col-span-2">
                <StatusBadge status={transcription.status} />
              </div>

              {/* Duration */}
              <div className="col-span-2 text-gray-300">
                {transcription.durationSeconds ? formatDuration(transcription.durationSeconds) : '-'}
              </div>

              {/* Date */}
              <div className="col-span-2 text-gray-400 text-sm">
                {formatDate(transcription.createdAt)}
              </div>

              {/* Actions */}
              <div className="col-span-2 flex items-center justify-end gap-2">
                {transcription.status === 'completed' && transcription.transcriptText && (
                  <ExportDropdown transcription={transcription} />
                )}
                <motion.button
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  onClick={() => {
                    if (confirm('Sei sicuro di voler eliminare questa trascrizione?')) {
                      onDelete(transcription.id);
                    }
                  }}
                  className="p-2 hover:bg-red-500/10 rounded-lg text-red-400 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </motion.button>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
