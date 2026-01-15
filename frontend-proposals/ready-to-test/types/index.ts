export interface Transcription {
  id: string;
  userId: string;
  fileName: string;
  fileSize: number;
  filePath: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  language?: string | null;
  durationSeconds?: number | null;
  transcriptText?: string | null;
  segments?: TranscriptionSegment[] | null;
  speakers?: SpeakerInfo | null;
  errorMessage?: string | null;
  createdAt: string;
  processingStartedAt?: string | null;
  processedAt?: string | null;
  geminiDocumentId?: string | null;
  detectedLanguage?: string | null;
  speakerCount?: number | null;
}

export interface TranscriptionSegment {
  start: number;
  end: number;
  text: string;
  speaker?: string | null;
  words?: Word[];
}

export interface Word {
  word: string;
  start: number;
  end: number;
  score?: number;
}

export interface SpeakerInfo {
  count: number;
  labels: string[];
}

export interface UsageStats {
  totalMinutes: number;
  totalCost: number;
  totalTranscriptions: number;
  byLanguage: Record<string, number>;
  byStatus: Record<string, number>;
}
