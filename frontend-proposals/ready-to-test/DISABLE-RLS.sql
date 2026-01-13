-- ================================================
-- DISABLE RLS FOR TESTING (NO AUTHENTICATION)
-- ================================================
--
-- Run this SQL in your Supabase SQL Editor to allow
-- the dashboard to work WITHOUT user authentication.
--
-- ⚠️ WARNING: This makes ALL data publicly accessible!
-- Only use for testing/development, NOT production!
--
-- ================================================

-- 1. Disable RLS on transcripts table
ALTER TABLE public.transcripts DISABLE ROW LEVEL SECURITY;

-- 2. Remove existing RLS policies (optional, for cleanup)
DROP POLICY IF EXISTS "Users can view own transcripts" ON public.transcripts;
DROP POLICY IF EXISTS "Users can insert own transcripts" ON public.transcripts;
DROP POLICY IF EXISTS "Users can update own transcripts" ON public.transcripts;
DROP POLICY IF EXISTS "Users can delete own transcripts" ON public.transcripts;

-- 3. Make storage bucket publicly accessible
-- Go to Storage → audio-temp → Settings → Make Public
-- Or run: UPDATE storage.buckets SET public = true WHERE name = 'audio-temp';

-- ================================================
-- DONE! Now the dashboard works without login
-- ================================================
