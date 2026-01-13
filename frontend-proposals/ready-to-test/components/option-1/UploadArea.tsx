'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

// Formati audio supportati da WhisperX
const SUPPORTED_FORMATS = [
  'audio/mpeg', // mp3
  'audio/mp4', // m4a
  'audio/x-m4a', // m4a alternative
  'audio/wav', // wav
  'audio/wave', // wav alternative
  'audio/x-wav', // wav alternative
  'audio/flac', // flac
  'audio/x-flac', // flac alternative
  'audio/ogg', // ogg
  'audio/webm', // webm
  'audio/aac', // aac
];

const SUPPORTED_EXTENSIONS = [
  '.mp3',
  '.m4a',
  '.wav',
  '.flac',
  '.ogg',
  '.webm',
  '.aac',
];

interface UploadAreaProps {
  onUpload: (file: File) => Promise<void>;
}

export function UploadArea({ onUpload }: UploadAreaProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      setError(null);

      if (acceptedFiles.length === 0) {
        setError('Formato file non supportato');
        return;
      }

      const file = acceptedFiles[0];

      // Validate file size (max 500MB)
      const maxSize = 500 * 1024 * 1024; // 500MB
      if (file.size > maxSize) {
        setError('File troppo grande (massimo 500MB)');
        return;
      }

      try {
        setIsUploading(true);
        await onUpload(file);
      } catch (err: any) {
        setError(err.message || 'Errore durante l\'upload');
      } finally {
        setIsUploading(false);
      }
    },
    [onUpload]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': SUPPORTED_EXTENSIONS,
    },
    multiple: false,
    disabled: isUploading,
  });

  return (
    <div>
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
          transition-all duration-200
          ${isDragActive
            ? 'border-violet-500 bg-violet-500/5'
            : 'border-zinc-700 bg-zinc-900/30 hover:border-zinc-600 hover:bg-zinc-900/50'
          }
          ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />

        {isUploading ? (
          <div className="flex flex-col items-center gap-4">
            <div className="w-12 h-12 rounded-full border-4 border-violet-500/20 border-t-violet-500 animate-spin" />
            <p className="text-zinc-400">Caricamento in corso...</p>
          </div>
        ) : (
          <>
            <div className="mx-auto w-16 h-16 rounded-full bg-zinc-800/50 flex items-center justify-center mb-4">
              <svg
                className="w-8 h-8 text-zinc-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>

            {isDragActive ? (
              <p className="text-lg text-violet-400 font-medium">
                Rilascia il file qui...
              </p>
            ) : (
              <>
                <p className="text-lg text-zinc-200 font-medium mb-2">
                  Trascina un file audio qui
                </p>
                <p className="text-sm text-zinc-400 mb-4">
                  oppure clicca per selezionare
                </p>
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-violet-500/10 border border-violet-500/20 rounded-lg text-sm text-violet-400">
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
                      d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"
                    />
                  </svg>
                  MP3, M4A, WAV, FLAC, OGG, WebM
                </div>
              </>
            )}
          </>
        )}
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-3">
          <svg
            className="w-5 h-5 text-red-400 flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Info */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="flex items-start gap-3 p-4 bg-zinc-900/30 rounded-lg border border-zinc-800/50">
          <div className="w-8 h-8 rounded-full bg-blue-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
            <svg
              className="w-4 h-4 text-blue-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-zinc-200">Veloce</p>
            <p className="text-xs text-zinc-400 mt-1">
              70x tempo reale con GPU
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3 p-4 bg-zinc-900/30 rounded-lg border border-zinc-800/50">
          <div className="w-8 h-8 rounded-full bg-violet-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
            <svg
              className="w-4 h-4 text-violet-400"
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
          </div>
          <div>
            <p className="text-sm font-medium text-zinc-200">Diarizzazione</p>
            <p className="text-xs text-zinc-400 mt-1">
              Riconoscimento speaker automatico
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3 p-4 bg-zinc-900/30 rounded-lg border border-zinc-800/50">
          <div className="w-8 h-8 rounded-full bg-emerald-500/10 flex items-center justify-center flex-shrink-0 mt-0.5">
            <svg
              className="w-4 h-4 text-emerald-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129"
              />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-zinc-200">Multilingua</p>
            <p className="text-xs text-zinc-400 mt-1">
              Supporto 90+ lingue
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
