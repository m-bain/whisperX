'use client';

import { useState, useRef, useEffect } from 'react';
import { exportTranscript } from '../utils/exportTranscript';
import type { Transcription } from '../types';

interface ExportDropdownProps {
  transcription: Transcription;
}

export function ExportDropdown({ transcription }: ExportDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  const handleExport = async (format: 'txt' | 'srt' | 'vtt' | 'json') => {
    setIsExporting(true);
    setIsOpen(false);

    try {
      await exportTranscript(transcription, format);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Esportazione fallita. Riprova.');
    } finally {
      setIsExporting(false);
    }
  };

  const formats = [
    {
      value: 'txt' as const,
      label: 'Testo (TXT)',
      description: 'Solo testo trascritto',
      icon: (
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
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
      ),
    },
    {
      value: 'srt' as const,
      label: 'Sottotitoli (SRT)',
      description: 'Con timestamp',
      icon: (
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
            d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z"
          />
        </svg>
      ),
    },
    {
      value: 'vtt' as const,
      label: 'WebVTT',
      description: 'Per video web',
      icon: (
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
            d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
          />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
    },
    {
      value: 'json' as const,
      label: 'JSON',
      description: 'Dati completi',
      icon: (
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
            d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
          />
        </svg>
      ),
    },
  ];

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isExporting}
        className="px-3 py-1.5 bg-violet-500/10 hover:bg-violet-500/20 border border-violet-500/20 rounded-lg text-sm text-violet-400 transition-colors flex items-center gap-2 disabled:opacity-50"
        title="Esporta"
      >
        {isExporting ? (
          <>
            <div className="w-4 h-4 border-2 border-violet-400/30 border-t-violet-400 rounded-full animate-spin" />
            <span>Esportazione...</span>
          </>
        ) : (
          <>
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
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
              />
            </svg>
            <span>Esporta</span>
          </>
        )}
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl overflow-hidden z-20">
          <div className="p-2">
            <p className="text-xs font-medium text-zinc-400 px-2 py-1.5 uppercase tracking-wide">
              Formato esportazione
            </p>
          </div>
          <div className="border-t border-zinc-800">
            {formats.map((format) => (
              <button
                key={format.value}
                onClick={() => handleExport(format.value)}
                className="w-full px-4 py-3 flex items-start gap-3 hover:bg-zinc-800/50 transition-colors text-left"
              >
                <div className="w-8 h-8 rounded-lg bg-zinc-800/50 flex items-center justify-center flex-shrink-0 text-zinc-400">
                  {format.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-zinc-200">
                    {format.label}
                  </p>
                  <p className="text-xs text-zinc-400 mt-0.5">
                    {format.description}
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
