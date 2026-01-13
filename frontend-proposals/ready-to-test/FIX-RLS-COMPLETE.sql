-- ================================================
-- VERIFICA STATO RLS E RIPARA
-- ================================================

-- 1. Controlla se RLS è ancora attivo
SELECT schemaname, tablename, rowsecurity
FROM pg_tables
WHERE tablename = 'transcripts';

-- 2. Mostra tutte le policy esistenti
SELECT * FROM pg_policies WHERE tablename = 'transcripts';

-- 3. FORZA DISABILITAZIONE RLS
ALTER TABLE IF EXISTS public.transcripts DISABLE ROW LEVEL SECURITY;

-- 4. ELIMINA TUTTE LE POLICY
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN (SELECT policyname FROM pg_policies WHERE tablename = 'transcripts') LOOP
        EXECUTE format('DROP POLICY IF EXISTS %I ON public.transcripts', r.policyname);
    END LOOP;
END $$;

-- 5. Verifica bucket storage
SELECT name, public FROM storage.buckets WHERE name = 'audio-temp';

-- 6. Rendi pubblico il bucket
UPDATE storage.buckets SET public = true WHERE name = 'audio-temp';

-- 7. VERIFICA FINALE
SELECT
    'transcripts RLS disabled' as check_type,
    NOT rowsecurity as ok
FROM pg_tables
WHERE tablename = 'transcripts'
UNION ALL
SELECT
    'storage bucket public' as check_type,
    public as ok
FROM storage.buckets
WHERE name = 'audio-temp';
