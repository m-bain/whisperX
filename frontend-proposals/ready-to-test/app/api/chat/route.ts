import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const geminiApiKey = process.env.GEMINI_API_KEY!;

export async function POST(request: NextRequest) {
  try {
    const { transcriptId, question } = await request.json();

    if (!transcriptId || !question) {
      return NextResponse.json(
        { error: 'Missing transcriptId or question' },
        { status: 400 }
      );
    }

    // Get transcript from database with full segments
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

    // Build full transcript text with speaker labels
    const segments = transcript.segments || [];
    let fullText = '';

    for (const seg of segments) {
      const speaker = seg.speaker ? `[${seg.speaker}]` : '';
      const text = seg.text || '';
      fullText += `${speaker} ${text}\n`;
    }

    if (!fullText.trim()) {
      return NextResponse.json(
        { error: 'Transcript is empty' },
        { status: 400 }
      );
    }

    // Initialize Gemini client
    const genAI = new GoogleGenerativeAI(geminiApiKey);
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    // Build the prompt with transcript context
    const prompt = `You are an AI assistant that answers questions about the following transcript.

TRANSCRIPT:
${fullText}

USER QUESTION: ${question}

Please provide a detailed and accurate answer based ONLY on the information in the transcript above. If the transcript doesn't contain information to answer the question, say so clearly.`;

    // Query Gemini with transcript text in prompt
    const result = await model.generateContent(prompt);

    const response = result.response;
    const answer = response.text();

    // Citations not available with direct prompt (would need RAG setup)
    const citations: any[] = [];

    return NextResponse.json({
      answer,
      citations,
      metadata: {
        fileName: transcript.fileName,
        language: transcript.language,
      },
    });
  } catch (error: any) {
    console.error('Chat API error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to process question' },
      { status: 500 }
    );
  }
}
