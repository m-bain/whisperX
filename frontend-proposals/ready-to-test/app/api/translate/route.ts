import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const geminiApiKey = process.env.GEMINI_API_KEY!;

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

export async function POST(request: NextRequest) {
  try {
    const { transcriptId, targetLanguage } = await request.json();

    if (!transcriptId || !targetLanguage) {
      return NextResponse.json(
        { error: 'Missing transcriptId or targetLanguage' },
        { status: 400 }
      );
    }

    if (!SUPPORTED_LANGUAGES[targetLanguage as keyof typeof SUPPORTED_LANGUAGES]) {
      return NextResponse.json(
        { error: `Unsupported language: ${targetLanguage}. Supported: ${Object.keys(SUPPORTED_LANGUAGES).join(', ')}` },
        { status: 400 }
      );
    }

    // Get transcript from database
    const supabase = createClient(supabaseUrl, supabaseKey);
    const { data: transcript, error } = await supabase
      .from('transcripts')
      .select('*')
      .eq('id', transcriptId)
      .single();

    if (error || !transcript) {
      return NextResponse.json(
        { error: 'Transcript not found' },
        { status: 404 }
      );
    }

    if (transcript.status !== 'completed') {
      return NextResponse.json(
        { error: 'Transcript not completed yet' },
        { status: 400 }
      );
    }

    // Check if already translated to target language
    const sourceLanguage = transcript.language;
    if (sourceLanguage === targetLanguage) {
      return NextResponse.json(
        { error: 'Transcript is already in the target language' },
        { status: 400 }
      );
    }

    // Initialize Gemini
    const genAI = new GoogleGenerativeAI(geminiApiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    // Prepare transcript text with speaker labels
    const segments = transcript.segments || [];
    let fullText = '';

    if (segments.length > 0) {
      for (const segment of segments) {
        const speaker = segment.speaker ? `[${segment.speaker}] ` : '';
        fullText += `${speaker}${segment.text}\n`;
      }
    } else {
      fullText = transcript.transcriptText || '';
    }

    // Translate using Gemini
    const targetLangName = SUPPORTED_LANGUAGES[targetLanguage as keyof typeof SUPPORTED_LANGUAGES];
    const prompt = `Translate the following transcript from ${sourceLanguage} to ${targetLangName}.

IMPORTANT INSTRUCTIONS:
- Maintain the speaker labels (e.g., [SPEAKER_00]) exactly as they are
- Preserve timestamps and formatting
- Translate only the spoken content, not the labels
- Keep natural flow and context
- DO NOT add any explanations or comments, only provide the translated text

TRANSCRIPT TO TRANSLATE:
${fullText}

TRANSLATED TRANSCRIPT:`;

    const result = await model.generateContent(prompt);
    const translatedText = result.response.text();

    // Parse translated text back into segments if possible
    const translatedSegments = [];
    const lines = translatedText.split('\n').filter(line => line.trim());

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const speakerMatch = line.match(/^\[([^\]]+)\]\s*(.+)$/);

      if (speakerMatch) {
        // Line has speaker label
        translatedSegments.push({
          speaker: speakerMatch[1],
          text: speakerMatch[2].trim(),
          index: i,
        });
      } else if (segments[i]) {
        // Use original segment structure
        translatedSegments.push({
          speaker: segments[i].speaker || null,
          text: line.trim(),
          index: i,
        });
      } else {
        // Fallback: no speaker
        translatedSegments.push({
          speaker: null,
          text: line.trim(),
          index: i,
        });
      }
    }

    return NextResponse.json({
      success: true,
      translation: {
        sourceLanguage,
        targetLanguage,
        sourceLangName: sourceLanguage.toUpperCase(),
        targetLangName,
        fullText: translatedText,
        segments: translatedSegments,
        fileName: transcript.fileName,
        wordCount: translatedText.split(/\s+/).length,
      },
    });
  } catch (error: any) {
    console.error('Translation API error:', error);
    return NextResponse.json(
      { error: error.message || 'Translation failed' },
      { status: 500 }
    );
  }
}
