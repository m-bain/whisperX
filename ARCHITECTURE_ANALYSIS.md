# Analisi Architettura: Centralizzare su Supabase

## 📊 Stack Attuale

| Servizio | Provider | Uso | Costo/Mese |
|----------|----------|-----|------------|
| **Database** | Supabase | PostgreSQL + Realtime | Free tier (500MB) |
| **Storage** | Supabase | File audio + trascrizioni | Free tier (1GB) |
| **Auth** | Supabase | Autenticazione utenti | Free tier |
| **GPU Processing** | Modal | WhisperX transcription | ~$10-50 (pay-per-use) |
| **Frontend Hosting** | ? (Vercel/Netlify) | Next.js dashboard | Free tier o ~$20/mese |
| **API Backend** | Next.js API Routes | Chat, Translate, Summarize | Incluso in hosting |
| **AI Models** | Google Gemini | RAG, traduzione, riassunti | Pay-per-use (~$0.50/mese) |

**Costo Totale Stimato:** $10-70/mese (dipende da usage)

---

## 🚀 Cosa Offre Supabase (Oltre a DB/Storage/Auth)

### **1. Supabase Edge Functions** (Deno runtime)
- **Cosa sono:** Serverless functions globalmente distribuite
- **Runtime:** Deno (TypeScript/JavaScript)
- **Pricing:**
  - Free tier: 500K invocations/mese, 400K GB-seconds
  - Pro: 2M invocations/mese, 1600K GB-seconds
- **Quando usare:** API endpoints, webhooks, scheduled jobs

### **2. Supabase Realtime**
- **Cosa è:** WebSocket connections per database changes
- **Già in uso?** Probabilmente sì (se usi subscriptions nel frontend)
- **Pricing:** Incluso nel piano DB

### **3. Supabase Vector (pgvector)**
- **Cosa è:** Embeddings e similarity search direttamente in PostgreSQL
- **Uso potenziale:** RAG avanzato (alternative a Gemini File Search)
- **Pricing:** Incluso, paga solo storage

### **4. Supabase Storage con Image Transformations**
- **Cosa è:** Resize, crop, optimize immagini on-the-fly
- **Uso:** Ottimizzazione file audio (transcoding?)
- **Pricing:** Pay-per-transform

### **5. Supabase API Auto-generated (PostgREST)**
- **Cosa è:** REST API automatica dal database schema
- **Già in uso?** SÌ (quando fai supabase.from('transcripts').select())
- **Pricing:** Incluso

---

## 🎯 Architettura Proposta: "Supabase-First"

### **Migrazione Possibile:**

| Servizio Attuale | Migrazione a Supabase | Convenienza |
|------------------|------------------------|-------------|
| **Next.js API Routes** (/api/chat, /api/translate) | ✅ Supabase Edge Functions | ⚠️ MEDIO |
| **Modal Worker (GPU)** | ❌ NON POSSIBILE | ❌ Modal è necessario per GPU |
| **Vercel Hosting** | ✅ Supabase Hosting (beta) | ⚠️ BASSO |
| **Gemini File Search** | ✅ Supabase Vector (pgvector) | 🤔 POSSIBILE ma complesso |

---

## ✅ **OPZIONE 1: Architettura "Supabase-Centric" (CONSIGLIATA)**

### **Stack:**
```
┌─────────────────────────────────────────────┐
│  Frontend: Next.js (Vercel/Netlify)         │
│  - Dashboard UI                             │
│  - File upload                              │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌───────────────┐      ┌────────────────────┐
│  Supabase     │      │  Modal (GPU only)  │
│               │      │                    │
│ • Database    │      │ • WhisperX         │
│ • Storage     │      │ • Diarization      │
│ • Auth        │      │ • Gemini upload    │
│ • Edge Funcs  │◄─────┤   (webhook)        │
│               │      └────────────────────┘
│ • Realtime    │
│ • Vector DB   │◄──── Gemini API
└───────────────┘      (traduzione, riassunti)
```

### **Cosa Migriamo:**

#### **1. Edge Functions per API Backend**
Spostiamo `/api/chat`, `/api/translate`, `/api/summarize` su Supabase Edge Functions:

```typescript
// supabase/functions/chat/index.ts
import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { GoogleGenerativeAI } from "npm:@google/generative-ai"

serve(async (req) => {
  const { transcriptId, question } = await req.json()

  // Query Supabase database (stesso server!)
  const { data: transcript } = await supabaseClient
    .from('transcripts')
    .select('geminiDocumentId')
    .eq('id', transcriptId)
    .single()

  // Call Gemini
  const genAI = new GoogleGenerativeAI(Deno.env.get('GEMINI_API_KEY')!)
  const result = await genAI.generateContent(...)

  return new Response(JSON.stringify({ answer }))
})
```

**Vantaggi:**
- ✅ Stesso datacenter del database (latenza <1ms)
- ✅ Autenticazione integrata (Supabase Auth)
- ✅ Environment variables gestite da Supabase
- ✅ Logs centralizzati

**Svantaggi:**
- ⚠️ Runtime Deno (non Node.js) - leggera curva di apprendimento
- ⚠️ Deploy separato (non automatico con Next.js)

---

#### **2. Supabase Realtime per Status Updates**
Già probabilmente in uso, ma ottimizziamo:

```typescript
// Frontend
const subscription = supabase
  .channel('transcripts')
  .on('postgres_changes',
    { event: 'UPDATE', schema: 'public', table: 'transcripts' },
    (payload) => {
      // Update UI in real-time quando Modal completa trascrizione
      console.log('Transcript updated:', payload.new)
    }
  )
  .subscribe()
```

**Vantaggio:** Niente polling, update istantanei.

---

#### **3. Supabase Vector per RAG Avanzato (OPZIONALE)**
Alternativa a Gemini File Search:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embeddings column
ALTER TABLE transcripts
ADD COLUMN embedding vector(1536);

-- Create index
CREATE INDEX ON transcripts
USING ivfflat (embedding vector_cosine_ops);
```

**Pro:**
- ✅ Nessun costo API esterna (Gemini File Search)
- ✅ Full control sui dati
- ✅ Similarity search velocissima

**Contro:**
- ❌ Devi generare embeddings (OpenAI/Cohere API o local model)
- ❌ Niente grounding metadata automatico (citazioni)
- ❌ Più complesso da implementare

**DECISIONE:** Teniamo Gemini File Search per ora (più semplice).

---

## 💰 **Confronto Costi**

### **Architettura Attuale (Next.js + Modal + Gemini)**
```
Supabase Free:           $0/mese
Modal (GPU):             $10-50/mese
Vercel (Next.js):        $0-20/mese
Gemini API:              ~$0.50/mese
────────────────────────────────
TOTALE:                  $10-70/mese
```

### **Architettura Supabase-Centric (Edge Functions)**
```
Supabase Pro:            $25/mese (include tutto: DB, Storage, Edge Functions)
Modal (GPU solo):        $10-50/mese (invariato)
Gemini API:              ~$0.50/mese
────────────────────────────────
TOTALE:                  $35-75/mese

MA con Free Tier Supabase + Edge Functions free tier:
────────────────────────────────
TOTALE:                  $10-50/mese (RISPARMIO!)
```

**NOTA:** Se rimani sotto i limiti free tier di Supabase Edge Functions (500K req/mese), **risparmi** perché non paghi Vercel hosting!

---

## 🎯 **RACCOMANDAZIONE FINALE**

### **Fase 1: Attuale (Mantieni così per ora)**
- ✅ Next.js API Routes per Chat/Translate/Summarize
- ✅ Modal per GPU
- ✅ Supabase per Database/Storage/Auth
- ✅ Vercel Free tier per hosting

**Costo:** $10-20/mese (free tier Vercel + Modal usage)

---

### **Fase 2: Ottimizzazione Futura (Quando scali)**
Quando superi 10K utenti/mese o 100K trascrizioni/mese:

1. **Migra API su Supabase Edge Functions** (Step-by-step)
   - Prima `/api/chat` → `supabase/functions/chat`
   - Poi `/api/translate` → `supabase/functions/translate`
   - Infine `/api/summarize` → `supabase/functions/summarize`

2. **Frontend rimane Next.js** (su Vercel o altro)

3. **Considera Supabase Vector** se:
   - Gemini API costa >$50/mese
   - Vuoi più controllo sul RAG

---

## 📋 **Piano di Migrazione (se decidi di farlo)**

### **Step 1: Setup Supabase CLI**
```bash
npm install supabase --save-dev
npx supabase init
```

### **Step 2: Crea Edge Function Test**
```bash
npx supabase functions new chat
```

### **Step 3: Deploy e Test**
```bash
npx supabase functions deploy chat --project-ref YOUR_PROJECT_REF
```

### **Step 4: Migra gradualmente**
- Frontend chiama Edge Function invece di `/api/chat`
- Testa throughput e latency
- Se OK, migra altre API

---

## ❓ **Cosa Ti Consiglio ORA**

**NON migrare ancora** su Supabase Edge Functions perché:

1. ✅ **Next.js API Routes funzionano benissimo** per il tuo uso
2. ✅ **Free tier Vercel** è sufficiente per testing/MVP
3. ✅ **Meno complessità** = più veloce andare in produzione
4. ⚠️ **Edge Functions Deno** = altra cosa da imparare

**QUANDO migrare:**
- 🚀 Quando superi free tier Vercel (100GB bandwidth/mese)
- 💰 Quando Vercel costa >$20/mese
- 📊 Quando hai >1000 utenti attivi/mese

---

## ✅ **DECISIONE PER OGGI**

**Teniamo architettura attuale:**
- Next.js API Routes (Frontend)
- Modal (GPU Worker)
- Supabase (Database + Storage + Auth)
- Gemini (AI features)

**E testiamo tutto per confermare che funziona!**

---

**Sei d'accordo con questa analisi?**

Vuoi che:
- ✅ **A) Procediamo con test del sistema attuale** (raccomandato)
- 🔄 **B) Migriamo subito su Edge Functions** (più tempo, ma ok se preferisci)
- 🤔 **C) Hai altre domande sull'architettura**
