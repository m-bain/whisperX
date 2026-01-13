'use client';

import Link from 'next/link';
import { UploadArea } from '../option-1/UploadArea';
import { TranscriptionTable } from '../option-1/TranscriptionTable';
import { useTranscriptions } from '../../hooks/useTranscriptions';
import { UsageChart } from './UsageChart';
import { CostTracker } from './CostTracker';
import { StatsCards } from './StatsCards';

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

  // Calculate stats
  const activeTranscriptions = transcriptions.filter(t => t.status === 'processing').length;
  const completedTranscriptions = transcriptions.filter(t => t.status === 'completed').length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800/50 backdrop-blur-sm sticky top-0 z-10 bg-slate-950/80">
        <div className="max-w-[1600px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/" className="text-slate-400 hover:text-slate-200 transition-colors">
                ← Home
              </Link>
              <div className="w-px h-6 bg-slate-800" />
              <div>
                <h1 className="text-xl font-semibold text-slate-100">
                  📊 Analytics Dashboard
                </h1>
                <p className="text-sm text-slate-400 mt-0.5">
                  Option 2: Visual Analytics
                </p>
              </div>
            </div>

            <button
              onClick={refreshTranscriptions}
              className="px-4 py-2 bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 rounded-lg text-sm text-violet-400 transition-colors"
            >
              Aggiorna
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-6 py-8">
        {/* Stats Cards */}
        <StatsCards
          totalMinutes={totalMinutes}
          totalCost={totalCost}
          activeTranscriptions={activeTranscriptions}
          completedTranscriptions={completedTranscriptions}
          totalTranscriptions={transcriptions.length}
        />

        {/* Dashboard Grid */}
        <div className="grid grid-cols-12 gap-6 mt-8">
          {/* Left Column - Charts */}
          <div className="col-span-12 lg:col-span-8 space-y-6">
            {/* Usage Chart */}
            <UsageChart transcriptions={transcriptions} />

            {/* Upload Area */}
            <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6 backdrop-blur-sm">
              <h3 className="text-lg font-semibold text-slate-100 mb-4">
                Carica Nuovo File
              </h3>
              <UploadArea onUpload={uploadFile} />
            </div>

            {/* Transcriptions Table */}
            <TranscriptionTable
              transcriptions={transcriptions}
              isLoading={isLoading}
              onDelete={deleteTranscription}
              onRefresh={refreshTranscriptions}
            />
          </div>

          {/* Right Column - Summary & Actions */}
          <div className="col-span-12 lg:col-span-4 space-y-6">
            <CostTracker
              totalCost={totalCost}
              totalMinutes={totalMinutes}
            />

            {/* Quick Stats */}
            <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6 backdrop-blur-sm">
              <h3 className="text-lg font-semibold text-slate-100 mb-4">
                Riepilogo Rapido
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between py-3 border-b border-slate-800/50">
                  <span className="text-sm text-slate-400">Totale File</span>
                  <span className="text-lg font-semibold text-slate-100">
                    {transcriptions.length}
                  </span>
                </div>
                <div className="flex items-center justify-between py-3 border-b border-slate-800/50">
                  <span className="text-sm text-slate-400">Completati</span>
                  <span className="text-lg font-semibold text-emerald-400">
                    {completedTranscriptions}
                  </span>
                </div>
                <div className="flex items-center justify-between py-3 border-b border-slate-800/50">
                  <span className="text-sm text-slate-400">In Elaborazione</span>
                  <span className="text-lg font-semibold text-violet-400">
                    {activeTranscriptions}
                  </span>
                </div>
                <div className="flex items-center justify-between py-3">
                  <span className="text-sm text-slate-400">Falliti</span>
                  <span className="text-lg font-semibold text-red-400">
                    {transcriptions.filter(t => t.status === 'failed').length}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
