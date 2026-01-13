import { formatSRTTimestamp, formatVTTTimestamp } from './formatters';
import type { Transcription, TranscriptionSegment } from '../types';

/**
 * Download a file with given content and filename
 */
function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Export as plain text
 */
function exportAsText(transcription: Transcription): string {
  if (transcription.transcriptText) {
    return transcription.transcriptText;
  }

  if (transcription.segments) {
    return transcription.segments
      .map((seg: TranscriptionSegment) => {
        const speaker = seg.speaker ? `[${seg.speaker}] ` : '';
        return `${speaker}${seg.text}`;
      })
      .join('\n\n');
  }

  return '';
}

/**
 * Export as SRT subtitles
 */
function exportAsSRT(transcription: Transcription): string {
  if (!transcription.segments || transcription.segments.length === 0) {
    throw new Error('No segments available for SRT export');
  }

  return transcription.segments
    .map((seg: TranscriptionSegment, index: number) => {
      const speaker = seg.speaker ? `[${seg.speaker}] ` : '';
      return [
        index + 1,
        `${formatSRTTimestamp(seg.start)} --> ${formatSRTTimestamp(seg.end)}`,
        `${speaker}${seg.text.trim()}`,
        '',
      ].join('\n');
    })
    .join('\n');
}

/**
 * Export as WebVTT
 */
function exportAsVTT(transcription: Transcription): string {
  if (!transcription.segments || transcription.segments.length === 0) {
    throw new Error('No segments available for VTT export');
  }

  const header = 'WEBVTT\n\n';
  const cues = transcription.segments
    .map((seg: TranscriptionSegment) => {
      const speaker = seg.speaker ? `<v ${seg.speaker}>` : '';
      return [
        `${formatVTTTimestamp(seg.start)} --> ${formatVTTTimestamp(seg.end)}`,
        `${speaker}${seg.text.trim()}`,
        '',
      ].join('\n');
    })
    .join('\n');

  return header + cues;
}

/**
 * Export as JSON
 */
function exportAsJSON(transcription: Transcription): string {
  const exportData = {
    metadata: {
      fileName: transcription.fileName,
      language: transcription.language,
      duration: transcription.durationSeconds,
      speakers: transcription.speakers,
      createdAt: transcription.createdAt,
      processedAt: transcription.processedAt,
    },
    transcript: transcription.transcriptText,
    segments: transcription.segments,
  };

  return JSON.stringify(exportData, null, 2);
}

/**
 * Main export function
 */
export async function exportTranscript(
  transcription: Transcription,
  format: 'txt' | 'srt' | 'vtt' | 'json'
) {
  const baseFilename = transcription.fileName.replace(/\.[^/.]+$/, '');

  let content: string;
  let filename: string;
  let mimeType: string;

  switch (format) {
    case 'txt':
      content = exportAsText(transcription);
      filename = `${baseFilename}.txt`;
      mimeType = 'text/plain';
      break;

    case 'srt':
      content = exportAsSRT(transcription);
      filename = `${baseFilename}.srt`;
      mimeType = 'text/plain';
      break;

    case 'vtt':
      content = exportAsVTT(transcription);
      filename = `${baseFilename}.vtt`;
      mimeType = 'text/vtt';
      break;

    case 'json':
      content = exportAsJSON(transcription);
      filename = `${baseFilename}.json`;
      mimeType = 'application/json';
      break;

    default:
      throw new Error(`Unsupported format: ${format}`);
  }

  downloadFile(content, filename, mimeType);
}
