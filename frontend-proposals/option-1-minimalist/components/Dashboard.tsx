'use client';

import { useState, useEffect } from 'react';
import { UploadArea } from './UploadArea';
import { TranscriptionTable } from './TranscriptionTable';
import { StatsHeader } from './StatsHeader';
import { useTranscriptions } from '../hooks/useTranscriptions';

export function Dashboard() {
  const {
    transcriptions,
    isLoading,
    totalMinutes,
    totalCost,
    uploadFile,
    deleteTranscription,
    refreshTranscriptions,
  } = useTranscriptions();

  // Auto-refresh every 5 seconds when there are processing items
  useEffect(() => {
    const hasProcessing = transcriptions.some(
      (t) => t.status === 'processing' || t.status === 'queued'
    );

    if (hasProcessing) {
      const interval = setInterval(refreshTranscriptions, 5000);
      return () => clearInterval(interval);
    }
  }, [transcriptions, refreshTranscriptions]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50">
      {/* Header */}
      <header className="border-b border-zinc-800/50 backdrop-blur-sm sticky top-0 z-10 bg-zinc-950/80">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold">Trascrizioni Audio</h1>
              <p className="text-sm text-zinc-400 mt-0.5">
                WhisperX - Trascrizioni con AI
              </p>
            </div>

            {/* User menu */}
            <div className="flex items-center gap-4">
              <button className="text-sm text-zinc-400 hover:text-zinc-50 transition-colors">
                Documentazione
              </button>
              <button className="text-sm text-zinc-400 hover:text-zinc-50 transition-colors">
                Impostazioni
              </button>
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500" />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Header */}
        <StatsHeader
          totalMinutes={totalMinutes}
          totalCost={totalCost}
          activeTranscriptions={
            transcriptions.filter((t) => t.status === 'processing').length
          }
        />

        {/* Upload Area */}
        <div className="mt-8">
          <UploadArea onUpload={uploadFile} />
        </div>

        {/* Transcriptions Table */}
        <div className="mt-8">
          <TranscriptionTable
            transcriptions={transcriptions}
            isLoading={isLoading}
            onDelete={deleteTranscription}
            onRefresh={refreshTranscriptions}
          />
        </div>
      </main>
    </div>
  );
}
