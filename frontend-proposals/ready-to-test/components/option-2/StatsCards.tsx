'use client';

interface StatsCardsProps {
  totalMinutes: number;
  totalCost: number;
  activeTranscriptions: number;
  completedTranscriptions: number;
  totalTranscriptions: number;
}

export function StatsCards({
  totalMinutes,
  totalCost,
  activeTranscriptions,
  completedTranscriptions,
  totalTranscriptions,
}: StatsCardsProps) {
  const completionRate = totalTranscriptions > 0
    ? ((completedTranscriptions / totalTranscriptions) * 100).toFixed(1)
    : '0';

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Total Minutes */}
      <div className="bg-gradient-to-br from-slate-900/80 to-slate-900/50 border border-slate-800/50 rounded-xl p-6 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-3">
          <div className="w-12 h-12 rounded-xl bg-violet-500/10 flex items-center justify-center">
            <svg className="w-6 h-6 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <span className="text-xs text-emerald-400 bg-emerald-500/10 px-2 py-1 rounded-md">
            +{Math.round((totalMinutes / 60) * 100) / 100} ore
          </span>
        </div>
        <p className="text-sm text-slate-400">Minuti Totali</p>
        <p className="text-3xl font-bold text-slate-100 mt-1">
          {totalMinutes.toFixed(1)}
        </p>
      </div>

      {/* Total Cost */}
      <div className="bg-gradient-to-br from-slate-900/80 to-slate-900/50 border border-slate-800/50 rounded-xl p-6 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-3">
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center">
            <svg className="w-6 h-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <span className="text-xs text-slate-500">
            €{(totalCost / (totalMinutes || 1)).toFixed(3)}/min
          </span>
        </div>
        <p className="text-sm text-slate-400">Costo Totale</p>
        <p className="text-3xl font-bold text-slate-100 mt-1">
          €{totalCost.toFixed(2)}
        </p>
      </div>

      {/* Completion Rate */}
      <div className="bg-gradient-to-br from-slate-900/80 to-slate-900/50 border border-slate-800/50 rounded-xl p-6 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-3">
          <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center">
            <svg className="w-6 h-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <span className="text-xs text-blue-400 bg-blue-500/10 px-2 py-1 rounded-md">
            {completionRate}%
          </span>
        </div>
        <p className="text-sm text-slate-400">Completati</p>
        <p className="text-3xl font-bold text-slate-100 mt-1">
          {completedTranscriptions}
        </p>
      </div>

      {/* Active Processing */}
      <div className="bg-gradient-to-br from-slate-900/80 to-slate-900/50 border border-slate-800/50 rounded-xl p-6 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-3">
          <div className="w-12 h-12 rounded-xl bg-orange-500/10 flex items-center justify-center">
            <svg className="w-6 h-6 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          {activeTranscriptions > 0 && (
            <div className="w-2 h-2 rounded-full bg-orange-400 animate-pulse" />
          )}
        </div>
        <p className="text-sm text-slate-400">In Elaborazione</p>
        <p className="text-3xl font-bold text-slate-100 mt-1">
          {activeTranscriptions}
        </p>
      </div>
    </div>
  );
}
