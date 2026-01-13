'use client';

import { useMemo } from 'react';
import type { Transcription } from '../../types';

interface UsageChartProps {
  transcriptions: Transcription[];
}

export function UsageChart({ transcriptions }: UsageChartProps) {
  // Group transcriptions by date
  const chartData = useMemo(() => {
    const last7Days = Array.from({ length: 7 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (6 - i));
      return date.toISOString().split('T')[0];
    });

    const dataByDate = last7Days.map(date => {
      const dayTranscriptions = transcriptions.filter(t => {
        const tDate = new Date(t.createdAt).toISOString().split('T')[0];
        return tDate === date;
      });

      const minutes = dayTranscriptions.reduce(
        (sum, t) => sum + ((t.durationSeconds || 0) / 60),
        0
      );

      return {
        date,
        minutes: Math.round(minutes * 10) / 10,
        count: dayTranscriptions.length,
      };
    });

    const maxMinutes = Math.max(...dataByDate.map(d => d.minutes), 1);

    return { dataByDate, maxMinutes };
  }, [transcriptions]);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('it-IT', { weekday: 'short', day: 'numeric' });
  };

  return (
    <div className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-6 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">
            Utilizzo nel Tempo
          </h3>
          <p className="text-sm text-slate-400 mt-1">
            Minuti trascritti negli ultimi 7 giorni
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-sm bg-violet-500" />
            <span className="text-slate-400">Minuti</span>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="space-y-6">
        {/* Bar Chart */}
        <div className="h-48 flex items-end gap-2">
          {chartData.dataByDate.map((day, i) => {
            const height = (day.minutes / chartData.maxMinutes) * 100;
            return (
              <div key={day.date} className="flex-1 flex flex-col items-center gap-2">
                <div className="w-full flex flex-col items-center justify-end h-full">
                  <div className="relative group w-full">
                    {/* Tooltip */}
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                      <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs whitespace-nowrap shadow-xl">
                        <p className="text-slate-100 font-medium">{day.minutes} min</p>
                        <p className="text-slate-400">{day.count} {day.count === 1 ? 'file' : 'file'}</p>
                      </div>
                    </div>

                    {/* Bar */}
                    <div
                      className="w-full bg-gradient-to-t from-violet-500 to-violet-400 rounded-t-lg transition-all duration-300 hover:from-violet-400 hover:to-violet-300"
                      style={{ height: `${Math.max(height, 4)}%` }}
                    />
                  </div>
                </div>

                {/* Label */}
                <span className="text-xs text-slate-500 text-center">
                  {formatDate(day.date)}
                </span>
              </div>
            );
          })}
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4 pt-6 border-t border-slate-800/50">
          <div>
            <p className="text-xs text-slate-400">Media Giornaliera</p>
            <p className="text-lg font-semibold text-slate-100 mt-1">
              {(chartData.dataByDate.reduce((sum, d) => sum + d.minutes, 0) / 7).toFixed(1)} min
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Picco Massimo</p>
            <p className="text-lg font-semibold text-slate-100 mt-1">
              {chartData.maxMinutes.toFixed(1)} min
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Totale 7 Giorni</p>
            <p className="text-lg font-semibold text-slate-100 mt-1">
              {chartData.dataByDate.reduce((sum, d) => sum + d.minutes, 0).toFixed(1)} min
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
