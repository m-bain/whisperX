'use client';

import { useState } from 'react';

interface SummarizationModalProps {
  transcriptId: string;
  fileName: string;
  onClose: () => void;
}

type SummaryType = 'brief' | 'detailed' | 'bullet-points' | 'executive';

interface SummaryResult {
  type: SummaryType;
  text: string;
  language: string;
  metadata: {
    fileName: string;
    sourceLanguage: string;
    duration: string;
    speakers: string[] | null;
    wordCount: number;
    summaryWordCount: number;
    created: string;
  };
}

const SUMMARY_TYPES = [
  {
    value: 'brief' as const,
    label: 'Breve',
    description: 'Riassunto di 2-3 frasi con i punti chiave',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
      </svg>
    ),
  },
  {
    value: 'detailed' as const,
    label: 'Dettagliato',
    description: 'Analisi completa con tutti i dettagli importanti',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
      </svg>
    ),
  },
  {
    value: 'bullet-points' as const,
    label: 'Punti Elenco',
    description: '5-10 punti chiave in formato elenco',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
      </svg>
    ),
  },
  {
    value: 'executive' as const,
    label: 'Esecutivo',
    description: 'Formato professionale con panoramica, scoperte e raccomandazioni',
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  },
];

export function SummarizationModal({ transcriptId, fileName, onClose }: SummarizationModalProps) {
  const [summaryType, setSummaryType] = useState<SummaryType>('brief');
  const [language, setLanguage] = useState('it');
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summary, setSummary] = useState<SummaryResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSummarize = async () => {
    setIsSummarizing(true);
    setError(null);

    try {
      const response = await fetch('/api/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transcriptId,
          summaryType,
          language,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Summarization failed');
      }

      setSummary(data.summary);
    } catch (err: any) {
      setError(err.message || 'Summarization failed');
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleDownload = () => {
    if (!summary) return;

    const content = `# ${SUMMARY_TYPES.find(t => t.value === summary.type)?.label} - ${summary.metadata.fileName}

**Tipo:** ${summary.type}
**Lingua:** ${summary.language}
**Durata:** ${summary.metadata.duration}
**Speaker:** ${summary.metadata.speakers ? summary.metadata.speakers.join(', ') : 'N/A'}
**Parole Originali:** ${summary.metadata.wordCount}
**Parole Riassunto:** ${summary.metadata.summaryWordCount}
**Creato:** ${new Date(summary.metadata.created).toLocaleString('it-IT')}

---

${summary.text}
`;

    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${fileName.replace(/\.[^.]+$/, '')}_${summary.type}_summary.md`;
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
            <h2 className="text-xl font-semibold text-white">Genera Riassunto</h2>
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
          {!summary ? (
            <div className="space-y-6">
              {/* Summary Type Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-3">
                  Tipo di Riassunto
                </label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {SUMMARY_TYPES.map((type) => (
                    <button
                      key={type.value}
                      onClick={() => setSummaryType(type.value)}
                      className={`p-4 rounded-lg border-2 transition-all text-left ${
                        summaryType === type.value
                          ? 'border-blue-500 bg-blue-500/10'
                          : 'border-gray-700 bg-gray-800 hover:border-gray-600'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <div className={`${summaryType === type.value ? 'text-blue-400' : 'text-gray-400'}`}>
                          {type.icon}
                        </div>
                        <div className="flex-1">
                          <div className={`font-medium ${summaryType === type.value ? 'text-blue-400' : 'text-white'}`}>
                            {type.label}
                          </div>
                          <div className="text-sm text-gray-400 mt-1">
                            {type.description}
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Language Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Lingua del Riassunto
                </label>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="it">Italiano</option>
                  <option value="en">English</option>
                </select>
              </div>

              {error && (
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
                  {error}
                </div>
              )}

              <button
                onClick={handleSummarize}
                disabled={isSummarizing}
                className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                {isSummarizing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Generazione in corso...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Genera Riassunto
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Summary Info */}
              <div className="flex items-center justify-between p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                <div className="flex items-center gap-2 text-green-400">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="font-medium">
                    Riassunto {SUMMARY_TYPES.find(t => t.value === summary.type)?.label} generato
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

              {/* Metadata */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                  <div className="text-xs text-gray-400">Durata</div>
                  <div className="text-sm font-medium text-white mt-1">{summary.metadata.duration}</div>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                  <div className="text-xs text-gray-400">Parole Originali</div>
                  <div className="text-sm font-medium text-white mt-1">{summary.metadata.wordCount}</div>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                  <div className="text-xs text-gray-400">Parole Riassunto</div>
                  <div className="text-sm font-medium text-white mt-1">{summary.metadata.summaryWordCount}</div>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                  <div className="text-xs text-gray-400">Riduzione</div>
                  <div className="text-sm font-medium text-green-400 mt-1">
                    {Math.round((1 - summary.metadata.summaryWordCount / summary.metadata.wordCount) * 100)}%
                  </div>
                </div>
              </div>

              {/* Summary Text */}
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
                <div className="prose prose-invert max-w-none">
                  <div className="whitespace-pre-wrap text-gray-200 leading-relaxed">
                    {summary.text}
                  </div>
                </div>
              </div>

              <button
                onClick={() => setSummary(null)}
                className="w-full px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white font-medium rounded-lg transition-colors"
              >
                Genera Nuovo Riassunto
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
