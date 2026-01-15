'use client';

import { motion } from 'framer-motion';
import { Transcription } from '../../types';
import { formatDuration, formatFileSize, formatDate } from '../../utils/formatters';
import { ExportDropdown } from '../option-1/ExportDropdown';

interface TranscriptionCardProps {
  transcription: Transcription;
  onDelete: (id: string) => void;
  index?: number;
}

export function TranscriptionCard({ transcription, onDelete, index = 0 }: TranscriptionCardProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'from-emerald-500/20 to-teal-500/20 border-emerald-500/30';
      case 'processing':
        return 'from-violet-500/20 to-purple-500/20 border-violet-500/30';
      case 'queued':
        return 'from-blue-500/20 to-cyan-500/20 border-blue-500/30';
      case 'failed':
        return 'from-red-500/20 to-orange-500/20 border-red-500/30';
      default:
        return 'from-gray-500/20 to-slate-500/20 border-gray-500/30';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return '✓';
      case 'processing':
        return '⟳';
      case 'queued':
        return '⏱';
      case 'failed':
        return '✗';
      default:
        return '?';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className={`bg-gradient-to-br ${getStatusColor(transcription.status)} rounded-xl border p-6 backdrop-blur-sm cursor-pointer group`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-gray-100 truncate mb-1 group-hover:text-emerald-300 transition-colors">
            {transcription.fileName}
          </h3>
          <p className="text-sm text-gray-400">
            {formatFileSize(transcription.fileSize)} • {formatDate(transcription.createdAt)}
          </p>
        </div>

        {/* Status Badge */}
        <div className="ml-3 flex-shrink-0">
          <motion.div
            whileHover={{ scale: 1.1 }}
            className={`w-10 h-10 rounded-lg flex items-center justify-center ${
              transcription.status === 'completed'
                ? 'bg-emerald-500/20 text-emerald-400'
                : transcription.status === 'processing'
                ? 'bg-violet-500/20 text-violet-400'
                : transcription.status === 'failed'
                ? 'bg-red-500/20 text-red-400'
                : 'bg-blue-500/20 text-blue-400'
            }`}
          >
            <span className="text-lg">{getStatusIcon(transcription.status)}</span>
          </motion.div>
        </div>
      </div>

      {/* Info Grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        {transcription.durationSeconds && (
          <div className="bg-gray-900/50 rounded-lg p-3">
            <p className="text-xs text-gray-500 mb-1">Durata</p>
            <p className="text-sm font-medium text-gray-200">
              {formatDuration(transcription.durationSeconds)}
            </p>
          </div>
        )}

        {transcription.detectedLanguage && (
          <div className="bg-gray-900/50 rounded-lg p-3">
            <p className="text-xs text-gray-500 mb-1">Lingua</p>
            <p className="text-sm font-medium text-gray-200 uppercase">
              {transcription.detectedLanguage}
            </p>
          </div>
        )}

        {transcription.speakerCount && transcription.speakerCount > 0 && (
          <div className="bg-gray-900/50 rounded-lg p-3">
            <p className="text-xs text-gray-500 mb-1">Speaker</p>
            <p className="text-sm font-medium text-gray-200">
              {transcription.speakerCount}
            </p>
          </div>
        )}

        <div className="bg-gray-900/50 rounded-lg p-3">
          <p className="text-xs text-gray-500 mb-1">Stato</p>
          <p className="text-sm font-medium capitalize text-gray-200">
            {transcription.status}
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 pt-4 border-t border-gray-700/50">
        {transcription.status === 'completed' && transcription.transcriptText && (
          <ExportDropdown transcription={transcription} />
        )}

        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={(e) => {
            e.stopPropagation();
            if (confirm('Sei sicuro di voler eliminare questa trascrizione?')) {
              onDelete(transcription.id);
            }
          }}
          className="ml-auto px-3 py-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg text-xs text-red-400 transition-colors"
        >
          Elimina
        </motion.button>
      </div>
    </motion.div>
  );
}
