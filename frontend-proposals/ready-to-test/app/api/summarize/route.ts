import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const geminiApiKey = process.env.GEMINI_API_KEY!;

type SummaryType = 'brief' | 'detailed' | 'bullet-points' | 'executive';

export async function POST(request: NextRequest) {
  try {
    const { transcriptId, summaryType = 'brief', language = 'it' } = await request.json();

    if (!transcriptId) {
      return NextResponse.json(
        { error: 'Missing transcriptId' },
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

    // Initialize Gemini
    const genAI = new GoogleGenerativeAI(geminiApiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    // Build prompt based on summary type
    let prompt = '';
    const outputLanguage = language === 'it' ? 'Italian' : language === 'en' ? 'English' : language;

    switch (summaryType) {
      case 'brief':
        prompt = `Create a brief summary (2-3 sentences) of the following transcript in ${outputLanguage}.

Focus on the main topic and key takeaway.

TRANSCRIPT:
${fullText}

BRIEF SUMMARY:`;
        break;

      case 'detailed':
        prompt = `Create a detailed summary of the following transcript in ${outputLanguage}.

Include:
- Main topics discussed
- Key points and arguments
- Important details and examples
- Conclusions or outcomes
- Speaker contributions (if multiple speakers)

TRANSCRIPT:
${fullText}

DETAILED SUMMARY:`;
        break;

      case 'bullet-points':
        prompt = `Create a bullet-point summary of the following transcript in ${outputLanguage}.

Format as:
- Main point 1
- Main point 2
- etc.

Include 5-10 key points that capture the essential information.

TRANSCRIPT:
${fullText}

BULLET-POINT SUMMARY:`;
        break;

      case 'executive':
        prompt = `Create an executive summary of the following transcript in ${outputLanguage}.

Include:
1. **Overview**: Brief context (1-2 sentences)
2. **Key Findings**: Main insights and discoveries
3. **Action Items**: Recommendations or next steps mentioned
4. **Conclusion**: Final takeaway

Format professionally suitable for stakeholders.

TRANSCRIPT:
${fullText}

EXECUTIVE SUMMARY:`;
        break;

      default:
        return NextResponse.json(
          { error: `Invalid summaryType: ${summaryType}. Use: brief, detailed, bullet-points, or executive` },
          { status: 400 }
        );
    }

    // Generate summary
    const result = await model.generateContent(prompt);
    const summaryText = result.response.text();

    // Extract metadata from transcript
    const speakers = transcript.speakers?.labels || [];
    const durationMinutes = Math.round((transcript.durationSeconds || 0) / 60);

    return NextResponse.json({
      success: true,
      summary: {
        type: summaryType,
        text: summaryText,
        language: outputLanguage,
        metadata: {
          fileName: transcript.fileName,
          sourceLanguage: transcript.language,
          duration: `${durationMinutes} minutes`,
          speakers: speakers.length > 0 ? speakers : null,
          wordCount: fullText.split(/\s+/).length,
          summaryWordCount: summaryText.split(/\s+/).length,
          created: new Date().toISOString(),
        },
      },
    });
  } catch (error: any) {
    console.error('Summarize API error:', error);
    return NextResponse.json(
      { error: error.message || 'Summarization failed' },
      { status: 500 }
    );
  }
}
