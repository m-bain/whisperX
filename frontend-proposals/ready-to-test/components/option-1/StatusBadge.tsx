'use client';

import { useEffect, useState } from 'react';
import { formatDuration } from '../../utils/formatters';

interface StatusBadgeProps {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  processingStartedAt?: string | null;
  processedAt?: string | null;
}

export function StatusBadge({
  status,
  processingStartedAt,
  processedAt,
}: StatusBadgeProps) {
  const [elapsedTime, setElapsedTime] = useState(0);

  // Calculate elapsed time for processing status
  useEffect(() => {
    if (status === 'processing' && processingStartedAt) {
      const interval = setInterval(() => {
        const elapsed = Math.floor(
          (Date.now() - new Date(processingStartedAt).getTime()) / 1000
        );
        setElapsedTime(elapsed);
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [status, processingStartedAt]);

  // Calculate total processing time for completed status
  const processingTime =
    status === 'completed' && processingStartedAt && processedAt
      ? Math.floor(
          (new Date(processedAt).getTime() -
            new Date(processingStartedAt).getTime()) /
            1000
        )
      : 0;

  const config = {
    queued: {
      color: 'text-blue-400',
      bg: 'bg-blue-500/10',
      border: 'border-blue-500/20',
      icon: (
        <svg
          className="w-3 h-3"
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
      ),
      label: 'In coda',
    },
    processing: {
      color: 'text-violet-400',
      bg: 'bg-violet-500/10',
      border: 'border-violet-500/20',
      icon: (
        <div className="w-3 h-3 border-2 border-violet-400/30 border-t-violet-400 rounded-full animate-spin" />
      ),
      label: 'Elaborazione',
    },
    completed: {
      color: 'text-emerald-400',
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/20',
      icon: (
        <svg
          className="w-3 h-3"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M5 13l4 4L19 7"
          />
        </svg>
      ),
      label: 'Completata',
    },
    failed: {
      color: 'text-red-400',
      bg: 'bg-red-500/10',
      border: 'border-red-500/20',
      icon: (
        <svg
          className="w-3 h-3"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      ),
      label: 'Fallita',
    },
  };

  const { color, bg, border, icon, label } = config[status];

  return (
    <div className="flex flex-col gap-1">
      <div
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium ${color} ${bg} border ${border} w-fit`}
      >
        {icon}
        {label}
      </div>

      {/* Show elapsed time for processing */}
      {status === 'processing' && elapsedTime > 0 && (
        <span className="text-xs text-zinc-500">
          {formatDuration(elapsedTime)}
        </span>
      )}

      {/* Show total time for completed */}
      {status === 'completed' && processingTime > 0 && (
        <span className="text-xs text-zinc-500">
          Completata in {formatDuration(processingTime)}
        </span>
      )}
    </div>
  );
}
