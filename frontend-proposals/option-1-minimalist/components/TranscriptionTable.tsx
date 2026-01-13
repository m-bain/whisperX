'use client';

import { useState } from 'react';
import { StatusBadge } from './StatusBadge';
import { ExportDropdown } from './ExportDropdown';
import { formatDuration, formatDate, formatFileSize } from '../utils/formatters';
import type { Transcription } from '../types';

interface TranscriptionTableProps {
  transcriptions: Transcription[];
  isLoading: boolean;
  onDelete: (id: string) => Promise<void>;
  onRefresh: () => void;
}

export function TranscriptionTable({
  transcriptions,
  isLoading,
  onDelete,
  onRefresh,
}: TranscriptionTableProps) {
  const [sortBy, setSortBy] = useState<'date' | 'duration' | 'status'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [filter, setFilter] = useState<'all' | 'completed' | 'processing' | 'failed'>('all');

  // Sort and filter
  const filteredTranscriptions = transcriptions
    .filter((t) => {
      if (filter === 'all') return true;
      return t.status === filter;
    })
    .sort((a, b) => {
      let comparison = 0;
      if (sortBy === 'date') {
        comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
      } else if (sortBy === 'duration') {
        comparison = (a.durationSeconds || 0) - (b.durationSeconds || 0);
      } else if (sortBy === 'status') {
        comparison = a.status.localeCompare(b.status);
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

  const handleSort = (column: typeof sortBy) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('desc');
    }
  };

  if (isLoading && transcriptions.length === 0) {
    return (
      <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-lg p-12 text-center">
        <div className="w-12 h-12 mx-auto rounded-full border-4 border-zinc-700 border-t-violet-500 animate-spin mb-4" />
        <p className="text-zinc-400">Caricamento trascrizioni...</p>
      </div>
    );
  }

  if (transcriptions.length === 0) {
    return (
      <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-lg p-12 text-center">
        <div className="w-16 h-16 mx-auto rounded-full bg-zinc-800/50 flex items-center justify-center mb-4">
          <svg
            className="w-8 h-8 text-zinc-600"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
        </div>
        <p className="text-lg text-zinc-200 font-medium mb-2">
          Nessuna trascrizione
        </p>
        <p className="text-sm text-zinc-400">
          Carica un file audio per iniziare
        </p>
      </div>
    );
  }

  return (
    <div className="bg-zinc-900/30 border border-zinc-800/50 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800/50 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-lg font-semibold text-zinc-100">
            Trascrizioni
          </h2>
          <span className="text-sm text-zinc-400">
            {filteredTranscriptions.length} {filteredTranscriptions.length === 1 ? 'risultato' : 'risultati'}
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* Filter */}
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as typeof filter)}
            className="px-3 py-1.5 bg-zinc-800/50 border border-zinc-700 rounded-lg text-sm text-zinc-300 focus:outline-none focus:border-violet-500"
          >
            <option value="all">Tutte</option>
            <option value="completed">Completate</option>
            <option value="processing">In elaborazione</option>
            <option value="failed">Fallite</option>
          </select>

          {/* Refresh */}
          <button
            onClick={onRefresh}
            className="px-3 py-1.5 bg-zinc-800/50 hover:bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-zinc-300 transition-colors flex items-center gap-2"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
            Aggiorna
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-zinc-900/50 border-b border-zinc-800/50">
            <tr>
              <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('date')}
                  className="flex items-center gap-1 hover:text-zinc-200 transition-colors"
                >
                  File
                  {sortBy === 'date' && (
                    <svg
                      className={`w-4 h-4 transition-transform ${
                        sortOrder === 'desc' ? 'rotate-180' : ''
                      }`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 15l7-7 7 7"
                      />
                    </svg>
                  )}
                </button>
              </th>
              <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('status')}
                  className="flex items-center gap-1 hover:text-zinc-200 transition-colors"
                >
                  Stato
                  {sortBy === 'status' && (
                    <svg
                      className={`w-4 h-4 transition-transform ${
                        sortOrder === 'desc' ? 'rotate-180' : ''
                      }`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 15l7-7 7 7"
                      />
                    </svg>
                  )}
                </button>
              </th>
              <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
                <button
                  onClick={() => handleSort('duration')}
                  className="flex items-center gap-1 hover:text-zinc-200 transition-colors"
                >
                  Durata
                  {sortBy === 'duration' && (
                    <svg
                      className={`w-4 h-4 transition-transform ${
                        sortOrder === 'desc' ? 'rotate-180' : ''
                      }`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 15l7-7 7 7"
                      />
                    </svg>
                  )}
                </button>
              </th>
              <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
                Lingua
              </th>
              <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
                Speaker
              </th>
              <th className="text-left px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
                Scadenza
              </th>
              <th className="text-right px-4 py-3 text-xs font-medium text-zinc-400 uppercase tracking-wider">
                Azioni
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {filteredTranscriptions.map((transcription) => (
              <TranscriptionRow
                key={transcription.id}
                transcription={transcription}
                onDelete={onDelete}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TranscriptionRow({
  transcription,
  onDelete,
}: {
  transcription: Transcription;
  onDelete: (id: string) => Promise<void>;
}) {
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDelete = async () => {
    if (!confirm('Sei sicuro di voler eliminare questa trascrizione?')) return;
    setIsDeleting(true);
    try {
      await onDelete(transcription.id);
    } catch (error) {
      console.error('Delete failed:', error);
      setIsDeleting(false);
    }
  };

  // Calculate expiration (30 days from creation)
  const expiresAt = new Date(transcription.createdAt);
  expiresAt.setDate(expiresAt.getDate() + 30);
  const daysUntilExpiration = Math.ceil(
    (expiresAt.getTime() - Date.now()) / (1000 * 60 * 60 * 24)
  );

  return (
    <tr className="hover:bg-zinc-900/50 transition-colors">
      <td className="px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-zinc-800/50 flex items-center justify-center flex-shrink-0">
            <svg
              className="w-5 h-5 text-zinc-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
              />
            </svg>
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium text-zinc-200 truncate">
              {transcription.fileName}
            </p>
            <p className="text-xs text-zinc-400 mt-0.5">
              {formatDate(transcription.createdAt)} • {formatFileSize(transcription.fileSize)}
            </p>
          </div>
        </div>
      </td>
      <td className="px-4 py-4">
        <StatusBadge
          status={transcription.status}
          processingStartedAt={transcription.processingStartedAt}
          processedAt={transcription.processedAt}
        />
      </td>
      <td className="px-4 py-4">
        <span className="text-sm text-zinc-300">
          {transcription.durationSeconds
            ? formatDuration(transcription.durationSeconds)
            : '-'}
        </span>
      </td>
      <td className="px-4 py-4">
        <span className="text-sm text-zinc-300">
          {transcription.language ? (
            <span className="inline-flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              {transcription.language.toUpperCase()}
            </span>
          ) : (
            '-'
          )}
        </span>
      </td>
      <td className="px-4 py-4">
        <span className="text-sm text-zinc-300">
          {transcription.speakers?.count ? (
            <span className="inline-flex items-center gap-1.5">
              <svg
                className="w-4 h-4 text-zinc-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                />
              </svg>
              {transcription.speakers.count}
            </span>
          ) : (
            '-'
          )}
        </span>
      </td>
      <td className="px-4 py-4">
        {daysUntilExpiration > 0 ? (
          <span
            className={`text-xs ${
              daysUntilExpiration <= 7
                ? 'text-orange-400'
                : 'text-zinc-400'
            }`}
          >
            {daysUntilExpiration} {daysUntilExpiration === 1 ? 'giorno' : 'giorni'}
          </span>
        ) : (
          <span className="text-xs text-red-400">Scaduto</span>
        )}
      </td>
      <td className="px-4 py-4">
        <div className="flex items-center justify-end gap-2">
          {transcription.status === 'completed' && (
            <ExportDropdown transcription={transcription} />
          )}
          <button
            onClick={handleDelete}
            disabled={isDeleting}
            className="p-2 hover:bg-zinc-800 rounded-lg transition-colors group disabled:opacity-50"
            title="Elimina"
          >
            {isDeleting ? (
              <div className="w-4 h-4 border-2 border-zinc-600 border-t-zinc-400 rounded-full animate-spin" />
            ) : (
              <svg
                className="w-4 h-4 text-zinc-400 group-hover:text-red-400 transition-colors"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            )}
          </button>
        </div>
      </td>
    </tr>
  );
}
