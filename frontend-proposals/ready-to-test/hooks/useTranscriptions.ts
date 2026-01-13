'use client';

import { useState, useEffect, useCallback } from 'react';
import { createClient } from '@supabase/supabase-js';
import type { Transcription } from '../types';

// Initialize Supabase client
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Modal webhook URL
const MODAL_WEBHOOK_URL = process.env.NEXT_PUBLIC_MODAL_WEBHOOK_URL!;

// Cost per minute (example: €0.01 per minute)
const COST_PER_MINUTE = 0.01;

export function useTranscriptions() {
  const [transcriptions, setTranscriptions] = useState<Transcription[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [totalMinutes, setTotalMinutes] = useState(0);
  const [totalCost, setTotalCost] = useState(0);

  // Fetch transcriptions
  const fetchTranscriptions = useCallback(async () => {
    try {
      const { data, error } = await supabase
        .from('transcripts')
        .select('*')
        .order('createdAt', { ascending: false });

      if (error) throw error;

      setTranscriptions(data || []);

      // Calculate total minutes and cost
      const minutes = (data || []).reduce(
        (sum, t) => sum + (t.durationSeconds || 0) / 60,
        0
      );
      setTotalMinutes(minutes);
      setTotalCost(minutes * COST_PER_MINUTE);
    } catch (error) {
      console.error('Error fetching transcriptions:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Upload and trigger transcription (NO AUTH REQUIRED)
  const uploadFile = useCallback(async (file: File) => {
    try {
      // Use a demo user ID (no authentication required)
      const demoUserId = 'demo-user';

      // 1. Upload file to Supabase Storage
      const filePath = `public/${Date.now()}-${file.name}`;
      const { error: uploadError } = await supabase.storage
        .from('audio-temp')
        .upload(filePath, file);

      if (uploadError) throw uploadError;

      // 2. Create transcript record in database
      const now = new Date().toISOString();
      const { data: transcript, error: dbError } = await supabase
        .from('transcripts')
        .insert({
          userId: demoUserId,
          fileName: file.name,
          fileSize: file.size,
          filePath,
          mimeType: file.type,
          status: 'queued',
          createdAt: now,
          updatedAt: now,
        })
        .select()
        .single();

      if (dbError) throw dbError;

      // 3. Trigger Modal transcription webhook (optional - skip if not configured)
      if (MODAL_WEBHOOK_URL && MODAL_WEBHOOK_URL !== 'your_modal_webhook_url_here') {
        try {
          const response = await fetch(MODAL_WEBHOOK_URL, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              transcript_id: transcript.id,
              file_path: filePath,
              user_id: demoUserId,
              language: null, // Auto-detect
              enable_diarization: true,
            }),
          });

          if (!response.ok) {
            console.warn(`Modal webhook failed: ${response.statusText}`);
            console.warn('File uploaded successfully but transcription will remain in "queued" status');
          }
        } catch (webhookError) {
          console.warn('Modal webhook error:', webhookError);
          console.warn('File uploaded successfully but transcription will remain in "queued" status');
        }
      } else {
        console.info('Modal webhook not configured - file will remain in "queued" status');
      }

      // 4. Refresh transcriptions list
      await fetchTranscriptions();
    } catch (error: any) {
      console.error('Upload error:', error);
      throw new Error(error.message || 'Upload failed');
    }
  }, [fetchTranscriptions]);

  // Delete transcription
  const deleteTranscription = useCallback(async (id: string) => {
    try {
      // Get transcript to find file path
      const { data: transcript } = await supabase
        .from('transcripts')
        .select('filePath')
        .eq('id', id)
        .single();

      if (transcript?.filePath) {
        // Delete file from storage
        await supabase.storage.from('audio-temp').remove([transcript.filePath]);
      }

      // Delete transcript record
      const { error } = await supabase.from('transcripts').delete().eq('id', id);
      if (error) throw error;

      // Refresh list
      await fetchTranscriptions();
    } catch (error) {
      console.error('Delete error:', error);
      throw error;
    }
  }, [fetchTranscriptions]);

  // Initial fetch
  useEffect(() => {
    fetchTranscriptions();
  }, [fetchTranscriptions]);

  // Subscribe to real-time updates
  useEffect(() => {
    const channel = supabase
      .channel('transcripts-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'transcripts',
        },
        () => {
          fetchTranscriptions();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [fetchTranscriptions]);

  return {
    transcriptions,
    isLoading,
    totalMinutes,
    totalCost,
    uploadFile,
    deleteTranscription,
    refreshTranscriptions: fetchTranscriptions,
  };
}
