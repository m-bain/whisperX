'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { useTranscriptions } from '../../hooks/useTranscriptions';
import { UploadArea } from '../option-1/UploadArea';
import { GridView } from './GridView';
import { ListView } from './ListView';
import { KanbanView } from './KanbanView';
import { ViewSwitcher, ViewMode } from './ViewSwitcher';
import { UserMenu } from '../UserMenu';

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

  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [showUpload, setShowUpload] = useState(false);

  // Calculate stats
  const completedCount = transcriptions.filter((t) => t.status === 'completed').length;
  const processingCount = transcriptions.filter((t) => t.status === 'processing').length;
  const failedCount = transcriptions.filter((t) => t.status === 'failed').length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900/20 to-gray-900">
      {/* Header */}
      <header className="border-b border-gray-800/50 backdrop-blur-sm sticky top-0 z-10 bg-gray-900/80">
        <div className="max-w-[1600px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="text-gray-400 hover:text-gray-200 transition-colors flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
                Home
              </Link>
              <div className="w-px h-6 bg-gray-800" />
              <div>
                <h1 className="text-xl font-semibold text-gray-100 flex items-center gap-2">
                  <span>🎴</span> Cards Dashboard
                </h1>
                <p className="text-sm text-gray-400 mt-0.5">Option 3: Card-based Modern</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <ViewSwitcher currentView={viewMode} onViewChange={setViewMode} />

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={refreshTranscriptions}
                className="p-2 bg-gray-800/50 hover:bg-gray-700/50 border border-gray-700/50 rounded-lg text-gray-400 hover:text-gray-200 transition-colors"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowUpload(!showUpload)}
                className="px-4 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 rounded-lg text-sm text-emerald-400 transition-colors font-medium"
              >
                + Carica File
              </motion.button>

              <UserMenu />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-6 py-8">
        {/* Stats Bar */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
        >
          <div className="bg-gradient-to-br from-gray-800/50 to-gray-800/30 border border-gray-700/50 rounded-xl p-4">
            <p className="text-sm text-gray-500 mb-1">Totale</p>
            <p className="text-2xl font-bold text-gray-100">{transcriptions.length}</p>
          </div>

          <div className="bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border border-emerald-500/30 rounded-xl p-4">
            <p className="text-sm text-emerald-400 mb-1">Completati</p>
            <p className="text-2xl font-bold text-emerald-300">{completedCount}</p>
          </div>

          <div className="bg-gradient-to-br from-violet-500/10 to-purple-500/10 border border-violet-500/30 rounded-xl p-4">
            <p className="text-sm text-violet-400 mb-1">In Elaborazione</p>
            <p className="text-2xl font-bold text-violet-300">{processingCount}</p>
          </div>

          <div className="bg-gradient-to-br from-red-500/10 to-orange-500/10 border border-red-500/30 rounded-xl p-4">
            <p className="text-sm text-red-400 mb-1">Falliti</p>
            <p className="text-2xl font-bold text-red-300">{failedCount}</p>
          </div>
        </motion.div>

        {/* Upload Area (Collapsible) */}
        <AnimatePresence>
          {showUpload && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-8 overflow-hidden"
            >
              <div className="bg-gray-800/30 border border-gray-700/50 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-100">Carica Nuovo File Audio</h3>
                  <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => setShowUpload(false)}
                    className="text-gray-400 hover:text-gray-200"
                  >
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </motion.button>
                </div>
                <UploadArea onUpload={uploadFile} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Views */}
        <AnimatePresence mode="wait">
          <motion.div
            key={viewMode}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {viewMode === 'grid' && (
              <GridView
                transcriptions={transcriptions}
                isLoading={isLoading}
                onDelete={deleteTranscription}
              />
            )}

            {viewMode === 'list' && (
              <ListView
                transcriptions={transcriptions}
                isLoading={isLoading}
                onDelete={deleteTranscription}
              />
            )}

            {viewMode === 'kanban' && (
              <KanbanView
                transcriptions={transcriptions}
                isLoading={isLoading}
                onDelete={deleteTranscription}
              />
            )}
          </motion.div>
        </AnimatePresence>

        {/* Footer Stats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-8 flex items-center justify-center gap-8 text-sm text-gray-500"
        >
          <div>
            Minuti totali: <span className="text-gray-300 font-medium">{totalMinutes.toFixed(1)}</span>
          </div>
          <div>
            Costo totale: <span className="text-emerald-400 font-medium">€{totalCost.toFixed(2)}</span>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
