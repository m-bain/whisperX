# 🎯 Setup Gemini RAG (Q&A Feature) - Istruzioni Complete

**Implementazione completata!** Ora devi solo configurare i secrets e fare il deploy.

---

## ✅ COSA È STATO FATTO

1. ✅ Aggiunta dipendenza `google-generativeai` al modal worker
2. ✅ Implementato upload automatico a Gemini File Search dopo trascrizione
3. ✅ Salvataggio del `geminiDocumentId` nel database Supabase
4. ✅ Gestione errori (se upload fallisce, trascrizione continua senza RAG)
5. ✅ Creato file SQL migration per aggiungere colonna al database
6. ✅ Cambiato modello da `gemini-2.0-flash-exp` a `gemini-1.5-flash` (quota più alta)

**Commits:**
- `02ef983` - feat: add Gemini File Search RAG upload to modal worker
- `fa63a45` - feat: add SQL migration for geminiDocumentId column
- `50da115` - fix: change Gemini model from 2.0-flash-exp to 1.5-flash

---

## 📋 COSA DEVI FARE TU (3 PASSI)

### PASSO 1: Aggiungere il campo `geminiDocumentId` al database

Vai su **Supabase SQL Editor** e esegui:

```bash
# Apri: https://app.supabase.com/project/YOUR_PROJECT/sql/new

# Copia e incolla il contenuto di:
frontend-proposals/ready-to-test/ADD-GEMINI-DOCUMENT-ID.sql
```

Oppure esegui direttamente questo SQL:

```sql
-- Add geminiDocumentId column
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

-- Add index for faster queries
CREATE INDEX IF NOT EXISTS idx_transcripts_gemini_document_id
ON public.transcripts("geminiDocumentId");
```

**Risultato:** Vedrai "Column geminiDocumentId added successfully"

---

### PASSO 2: Configurare Gemini API Key come Modal Secret

**Se hai già il secret `gemini-api`:**
```bash
# Verifica se esiste
modal secret list

# Se esiste già "gemini-api", SKIP questo passo
```

**Se NON hai ancora il secret:**

```bash
# Crea il secret con la tua Gemini API Key
modal secret create gemini-api GEMINI_API_KEY=your_actual_gemini_api_key_here
```

**Come ottenere la Gemini API Key:**
1. Vai su https://aistudio.google.com/apikey
2. Clicca "Create API Key"
3. Copia la chiave
4. Usala nel comando sopra (sostituisci `your_actual_gemini_api_key_here`)

**Verifica:**
```bash
modal secret list
# Dovresti vedere:
# - gemini-api
# - supabase-credentials
```

---

### PASSO 3: Deploy del Modal Worker Aggiornato

```bash
cd C:\Users\nicol\whisperx

# 1. Pull del codice aggiornato
git pull origin claude/project-status-review-01HkxUm3YMNY6vWEEhXVSH5U

# 2. Deploy su Modal
modal deploy modal_worker.py
```

**Tempo di build:** ~5-10 minuti (deve installare google-generativeai)

---

## 🧪 TESTING

Dopo il deploy, testa l'intero flusso:

### 1. Carica un nuovo audio file
```
localhost:3000 → Upload audio → Attendi trascrizione
```

### 2. Verifica il pulsante Q&A
Nella tabella delle trascrizioni, dovresti vedere:
- ✅ 💬 (Chat Q&A) - **NUOVO! Prima non c'era**
- ✅ 🌐 (Translation)
- ✅ 📄 (Summarization)

### 3. Testa il Q&A
```
Clicca 💬 → Inserisci una domanda:
"Fai un riassunto della trascrizione"
"Quali sono i punti principali discussi?"
"Chi sono gli speaker e cosa hanno detto?"
```

Il sistema userà **Gemini File Search RAG** per rispondere!

### 4. Verifica nel database
```sql
SELECT id, fileName, status, "geminiDocumentId"
FROM transcripts
WHERE status = 'completed'
ORDER BY "processedAt" DESC
LIMIT 5;
```

Dovresti vedere il `geminiDocumentId` popolato (es: `files/abc123xyz`)

---

## 🔍 TROUBLESHOOTING

### Il pulsante 💬 non compare

**Causa:** Il `geminiDocumentId` è NULL nel database

**Fix:**
1. Controlla i log Modal per vedere se l'upload è fallito
2. Verifica che il secret `gemini-api` sia configurato
3. Verifica che GEMINI_API_KEY sia valida

### Errore "GEMINI_API_KEY not found"

**Causa:** Secret non configurato correttamente

**Fix:**
```bash
# Ricrea il secret
modal secret create gemini-api GEMINI_API_KEY=your_key_here --force
```

### Errore 429 "Quota exceeded"

**Causa:** Gemini free tier ha limiti bassi

**Fix:** Il codice usa già `gemini-1.5-flash` che ha quota alta. Se persiste:
1. Aspetta qualche minuto (rate limit)
2. Considera Gemini paid tier ($0.00035/1K tokens - economico!)

### Upload a Gemini fallisce ma trascrizione funziona

**Questo è normale!** Il codice è progettato per continuare senza RAG se upload fallisce.

**Risultato:**
- ✅ Trascrizione salvata
- ✅ Translation e Summarization funzionano
- ❌ Q&A non disponibile (nessun pulsante 💬)

---

## 📊 COME FUNZIONA IL FLUSSO

```
1. Audio Upload (Frontend)
         ↓
2. Modal Worker: WhisperX Transcription
         ↓
3. Salvataggio su Supabase Storage (JSON)
         ↓
4. Upload a Gemini File Search (RAG)
         ├─ Success → geminiDocumentId salvato in DB
         └─ Failed  → Continua senza RAG
         ↓
5. Database Update (status=completed + geminiDocumentId)
         ↓
6. Frontend mostra pulsanti:
   - 💬 (solo se geminiDocumentId presente)
   - 🌐 (sempre)
   - 📄 (sempre)
```

---

## 🎉 BENEFICI RAG

**Senza RAG (solo Gemini LLM):**
- ❌ Risposte generiche
- ❌ Nessun contesto dalla trascrizione
- ❌ Nessuna citazione

**Con RAG (Gemini File Search):**
- ✅ Risposte accurate basate sulla trascrizione
- ✅ Contesto completo
- ✅ Citazioni precise con riferimenti
- ✅ Può trovare informazioni specifiche nel testo

---

## 🔐 PRIVACY & DATA RETENTION

### Dove stanno i dati?

| Storage | Dati | Retention | Controllo |
|---------|------|-----------|-----------|
| **Supabase DB** | Metadati + 5000 caratteri | ∞ Permanente | 100% tuo |
| **Supabase Storage** | JSON completo | ∞ Permanente | 100% tuo |
| **Gemini File Search** | Copia per RAG | 48 ore* | Google |

\* I file su Gemini vengono auto-cancellati dopo 48 ore per default

### GDPR Compliance

Se hai utenti EU, devi:
1. ✅ Informare che i dati vengono processati da Google Gemini
2. ✅ Aggiungere alla privacy policy
3. ✅ Ottenere consenso esplicito per upload a Google

**Opzione:** Rendi il RAG Q&A **opt-in** invece di automatico.

---

## 🚀 PROSSIMI PASSI

Dopo aver completato i 3 passi sopra:

1. ✅ **Test completo** - Verifica che tutto funzioni
2. ✅ **Documentazione utente** - Spiega come usare Q&A
3. 🔄 **Opzionale:** Implementa pulizia automatica file Gemini vecchi
4. 🔄 **Opzionale:** Aggiungi retry logic per upload falliti
5. 🔄 **Opzionale:** Implementa cache delle risposte Q&A

---

## 📝 CHANGELOG

**18 Nov 2024 - v1.0**
- ✅ Implementato Gemini File Search RAG upload
- ✅ Aggiunto google-generativeai al modal worker
- ✅ Cambiato modello a gemini-1.5-flash (quota più alta)
- ✅ Creato SQL migration per geminiDocumentId
- ✅ Gestione errori graceful (fallback senza RAG)

---

**FINE SETUP INSTRUCTIONS** 🎯
