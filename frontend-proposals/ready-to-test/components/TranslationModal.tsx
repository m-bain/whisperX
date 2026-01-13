'use client';

import { useState } from 'react';

interface TranslationModalProps {
  transcriptId: string;
  fileName: string;
  currentLanguage: string;
  onClose: () => void;
}

const SUPPORTED_LANGUAGES = {
  'en': 'English',
  'it': 'Italian',
  'fr': 'French',
  'de': 'German',
  'es': 'Spanish',
  'pt': 'Portuguese',
  'ru': 'Russian',
  'zh': 'Chinese',
  'ja': 'Japanese',
  'ko': 'Korean',
  'ar': 'Arabic',
};

interface TranslationResult {
  sourceLanguage: string;
  targetLanguage: string;
  sourceLangName: string;
  targetLangName: string;
  fullText: string;
  segments: Array<{
    speaker: string | null;
    text: string;
    index: number;
  }>;
  fileName: string;
  wordCount: number;
}

export function TranslationModal({ transcriptId, fileName, currentLanguage, onClose }: TranslationModalProps) {
  const [targetLanguage, setTargetLanguage] = useState('en');
  const [isTranslating, setIsTranslating] = useState(false);
  const [translation, setTranslation] = useState<TranslationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTranslate = async () => {
    setIsTranslating(true);
    setError(null);

    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transcriptId,
          targetLanguage,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Translation failed');
      }

      setTranslation(data.translation);
    } catch (err: any) {
      setError(err.message || 'Translation failed');
    } finally {
      setIsTranslating(false);
    }
  };

  const handleDownload = () => {
    if (!translation) return;

    const blob = new Blob([translation.fullText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fileName.replace(/\.[^.]+$/, '')}_${translation.targetLanguage}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 border border-gray-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="border-b border-gray-800 p-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">Traduci Trascrizione</h2>
            <p className="text-sm text-gray-400 mt-1">{fileName}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1">
          {!translation ? (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Lingua Originale
                </label>
                <input
                  type="text"
                  value={currentLanguage.toUpperCase()}
                  disabled
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 cursor-not-allowed"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Traduci in
                </label>
                <select
                  value={targetLanguage}
                  onChange={(e) => setTargetLanguage(e.target.value)}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  {Object.entries(SUPPORTED_LANGUAGES)
                    .filter(([code]) => code !== currentLanguage)
                    .map(([code, name]) => (
                      <option key={code} value={code}>
                        {name}
                      </option>
                    ))}
                </select>
              </div>

              {error && (
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
                  {error}
                </div>
              )}

              <button
                onClick={handleTranslate}
                disabled={isTranslating}
                className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {isTranslating ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Traduzione in corso...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                    </svg>
                    Traduci
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Translation Info */}
              <div className="flex items-center justify-between p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                <div className="flex items-center gap-2 text-green-400">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="font-medium">
                    Tradotto da {translation.sourceLangName} a {translation.targetLangName}
                  </span>
                </div>
                <button
                  onClick={handleDownload}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg flex items-center gap-2 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Scarica
                </button>
              </div>

              {/* Translation Text */}
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
                <div className="text-sm text-gray-400 mb-2">
                  {translation.wordCount} parole
                </div>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {translation.segments.map((segment, index) => (
                    <div key={index} className="text-gray-200">
                      {segment.speaker && (
                        <span className="text-blue-400 font-medium mr-2">
                          [{segment.speaker}]
                        </span>
                      )}
                      <span>{segment.text}</span>
                    </div>
                  ))}
                </div>
              </div>

              <button
                onClick={() => setTranslation(null)}
                className="w-full px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white font-medium rounded-lg transition-colors"
              >
                Nuova Traduzione
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
