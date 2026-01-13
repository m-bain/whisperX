-- ================================================
-- ADD geminiDocumentId COLUMN FOR GEMINI FILE SEARCH RAG
-- ================================================
--
-- Run this SQL in your Supabase SQL Editor to add the
-- geminiDocumentId column to the transcripts table.
--
-- This column stores the Gemini File Search document ID
-- for enabling Q&A RAG functionality.
--
-- ================================================

-- Add geminiDocumentId column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'transcripts'
        AND column_name = 'geminiDocumentId'
    ) THEN
        ALTER TABLE public.transcripts
        ADD COLUMN "geminiDocumentId" TEXT NULL;

        RAISE NOTICE 'Column geminiDocumentId added successfully';
    ELSE
        RAISE NOTICE 'Column geminiDocumentId already exists';
    END IF;
END $$;

-- Add index for faster queries (optional but recommended)
CREATE INDEX IF NOT EXISTS idx_transcripts_gemini_document_id
ON public.transcripts("geminiDocumentId");

-- ================================================
-- DONE! The transcripts table now supports RAG Q&A
-- ================================================
