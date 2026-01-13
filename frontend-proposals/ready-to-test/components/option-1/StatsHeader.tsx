'use client';

interface StatsHeaderProps {
  totalMinutes: number;
  totalCost: number;
  activeTranscriptions: number;
}

export function StatsHeader({
  totalMinutes,
  totalCost,
  activeTranscriptions,
}: StatsHeaderProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {/* Total Minutes */}
      <div className="bg-zinc-900/50 border border-zinc-800/50 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-zinc-400">Minuti Totali</p>
            <p className="text-3xl font-semibold mt-1">
              {totalMinutes.toFixed(1)}
            </p>
          </div>
          <div className="w-12 h-12 rounded-full bg-violet-500/10 flex items-center justify-center">
            <svg
              className="w-6 h-6 text-violet-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
        </div>
        <p className="text-xs text-zinc-500 mt-2">
          {(totalMinutes / 60).toFixed(1)} ore processate
        </p>
      </div>

      {/* Total Cost */}
      <div className="bg-zinc-900/50 border border-zinc-800/50 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-zinc-400">Costo Totale</p>
            <p className="text-3xl font-semibold mt-1">
              €{totalCost.toFixed(2)}
            </p>
          </div>
          <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center">
            <svg
              className="w-6 h-6 text-emerald-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
        </div>
        <p className="text-xs text-zinc-500 mt-2">
          Media €{totalMinutes > 0 ? (totalCost / totalMinutes).toFixed(3) : '0.000'} / minuto
        </p>
      </div>

      {/* Active Transcriptions */}
      <div className="bg-zinc-900/50 border border-zinc-800/50 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-zinc-400">In Elaborazione</p>
            <p className="text-3xl font-semibold mt-1">{activeTranscriptions}</p>
          </div>
          <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center">
            <svg
              className="w-6 h-6 text-blue-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
        </div>
        {activeTranscriptions > 0 ? (
          <div className="flex items-center gap-2 mt-2">
            <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            <p className="text-xs text-zinc-500">Elaborazione in corso</p>
          </div>
        ) : (
          <p className="text-xs text-zinc-500 mt-2">Nessuna elaborazione attiva</p>
        )}
      </div>
    </div>
  );
}
