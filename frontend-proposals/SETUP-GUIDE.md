# 🚀 Guida Setup Completa - Dashboard WhisperX

## 📋 Panoramica

Hai a disposizione **3 dashboard complete** per il tuo sistema di trascrizione audio con WhisperX.

### ✅ Tutte le funzionalità richieste sono implementate

1. **✅ Supporto formati audio completo**
   - MP3, M4A, WAV, FLAC, OGG, WebM, AAC
   - Validazione file size (max 500MB)
   - Drag & drop con feedback visivo

2. **✅ Conteggio minuti e costi**
   - Tracking minuti totali trascritti
   - Calcolo costi automatico (€0.01/minuto - configurabile)
   - Statistiche in real-time

3. **✅ Gestione scadenza file**
   - File scadono dopo 30 giorni (configurabile)
   - Mostra giorni rimanenti
   - Warning quando scadenza vicina

4. **✅ Stato trascrizione real-time**
   - `queued` → In coda
   - `processing` → Elaborazione (con timer live che mostra quanto dura)
   - `completed` → Completata (mostra tempo totale elaborazione)
   - `failed` → Fallita (con messaggio errore)

5. **✅ Esportazione funzionante**
   - TXT (solo testo)
   - SRT (sottotitoli con timestamp)
   - VTT (WebVTT per video web)
   - JSON (dati completi con metadata)

6. **✅ Interfaccia moderna**
   - 3 opzioni di design completamente diverse
   - Dark mode
   - Responsive (desktop-first, mobile-ready)
   - Animazioni fluide

---

## 🎨 Quale Dashboard Scegliere?

### Option 1: Minimalist Professional ⭐ **CONSIGLIATA PER MVP**

**Quando sceglierla:**
- Vuoi partire veloce con un MVP
- Preferisci interfaccia pulita e professionale
- Ti servono solo le funzionalità essenziali
- Vuoi performance ottimali

**Pro:**
- Setup più rapido (meno dipendenze)
- Codice più semplice da mantenere
- Ottima performance
- Design professionale

**Contro:**
- Meno "wow factor"
- Pochi grafici/visualizzazioni

**File principali:**
- `Dashboard.tsx` - Layout principale
- `UploadArea.tsx` - Upload drag & drop
- `TranscriptionTable.tsx` - Tabella sortable/filterable
- `StatusBadge.tsx` - Indicatori stato con timer
- `ExportDropdown.tsx` - Menu esportazione

---

### Option 2: Visual Analytics 📊 **CONSIGLIATA PER BUSINESS**

**Quando sceglierla:**
- Serve reporting e analytics
- Vuoi dashboard "premium"
- Hai bisogno di insights dettagliati
- Target: business users o team

**Pro:**
- Ricchissima di dati e grafici
- Perfetta per presentazioni
- Tracking budget e proiezioni
- Look premium

**Contro:**
- Più complessa da setup
- Richiede più librerie (Recharts)
- Può essere overwhelming per utenti casual

**Features extra:**
- Usage Chart (trend temporale)
- Cost Tracker (budget + proiezioni)
- Language Distribution (pie chart)
- Timeline view
- Heatmap utilizzo orario

---

### Option 3: Card-based Modern 🎴 **CONSIGLIATA PER UX**

**Quando sceglierla:**
- Vuoi massimo engagement
- Design moderno e trendy è priorità
- Target: utenti consumer
- Vuoi flessibilità (grid/list/kanban)

**Pro:**
- Visivamente stunning
- Animazioni fluide (Framer Motion)
- Multiple views (grid/list/kanban)
- UX moderna

**Contro:**
- Più pesante (animazioni)
- Più complessa da mantenere
- Richiede più librerie

**Features extra:**
- Grid/List/Kanban toggle
- Drag & drop cards
- Modal full-screen
- Keyboard shortcuts
- Undo/Redo

---

## 🛠️ Setup Generale (Per Tutte le Opzioni)

### 1. Prerequisiti

```bash
# Node.js 18+
node --version

# Next.js 14+ project
npx create-next-app@latest my-transcription-app
cd my-transcription-app
```

### 2. Install Dependencies Base

```bash
# Dependencies comuni a tutte le opzioni
npm install @supabase/supabase-js react-dropzone

# Option 1: Nessuna dipendenza extra

# Option 2: Analytics charts
npm install recharts date-fns

# Option 3: Animations & DnD
npm install framer-motion react-beautiful-dnd
```

### 3. Environment Variables

Crea `.env.local`:

```env
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key

# Modal Webhook
NEXT_PUBLIC_MODAL_WEBHOOK_URL=https://your-modal-app.modal.run
```

### 4. Supabase Setup

#### A) Database Schema

```sql
-- Create transcripts table
create table public.transcripts (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users not null,
  file_name text not null,
  file_size bigint not null,
  file_path text not null,
  status text not null check (status in ('queued', 'processing', 'completed', 'failed')),
  language text,
  duration_seconds integer,
  transcript_text text,
  segments jsonb,
  speakers jsonb,
  error_message text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  processing_started_at timestamp with time zone,
  processed_at timestamp with time zone
);

-- Enable RLS
alter table public.transcripts enable row level security;

-- Policies
create policy "Users can view own transcripts"
  on public.transcripts for select
  using (auth.uid() = user_id);

create policy "Users can insert own transcripts"
  on public.transcripts for insert
  with check (auth.uid() = user_id);

create policy "Users can update own transcripts"
  on public.transcripts for update
  using (auth.uid() = user_id);

create policy "Users can delete own transcripts"
  on public.transcripts for delete
  using (auth.uid() = user_id);

-- Enable Realtime
alter publication supabase_realtime add table transcripts;

-- Indexes for performance
create index idx_transcripts_user_id on public.transcripts(user_id);
create index idx_transcripts_status on public.transcripts(status);
create index idx_transcripts_created_at on public.transcripts(created_at desc);
```

#### B) Storage Buckets

Crea 2 bucket in Supabase Storage:

**1. audio-temp** (per file audio caricati)

```sql
-- RLS Policy per audio-temp
create policy "Users can upload own audio"
  on storage.objects for insert
  with check (
    bucket_id = 'audio-temp' and
    auth.uid()::text = (storage.foldername(name))[1]
  );

create policy "Users can view own audio"
  on storage.objects for select
  using (
    bucket_id = 'audio-temp' and
    auth.uid()::text = (storage.foldername(name))[1]
  );

create policy "Users can delete own audio"
  on storage.objects for delete
  using (
    bucket_id = 'audio-temp' and
    auth.uid()::text = (storage.foldername(name))[1]
  );
```

**2. transcripts** (per JSON delle trascrizioni)

Stesse policy di `audio-temp` ma con `bucket_id = 'transcripts'`.

### 5. Modal Worker Configuration

Il `modal_worker.py` è già configurato correttamente con:

✅ Supporto formati: MP3, M4A, WAV, FLAC, OGG, WebM (via ffmpeg-python)
✅ pkg-config e FFmpeg dev libraries installate
✅ WhisperX 3.2.0 + ctranslate2 4.4.0
✅ PyTorch 2.0.0 + CUDA 11.8

**Setup Modal secrets:**

```bash
# Crea secret con credenziali Supabase
modal secret create supabase-credentials \
  SUPABASE_URL=https://your-project.supabase.co \
  SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

**Deploy Modal worker:**

```bash
modal deploy modal_worker.py
```

Ti verrà fornito un webhook URL tipo:
`https://your-username--whisperx-transcription-transcribe-webhook.modal.run`

Aggiungi questo URL al tuo `.env.local` come `NEXT_PUBLIC_MODAL_WEBHOOK_URL`.

---

## 📁 Struttura Progetto Finale

```
my-transcription-app/
├── app/
│   ├── dashboard/
│   │   └── page.tsx              # Usa uno dei Dashboard.tsx
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── Dashboard.tsx             # Copia da option-X/components/
│   ├── UploadArea.tsx
│   ├── Transcription*.tsx
│   ├── StatusBadge.tsx
│   └── ExportDropdown.tsx
├── hooks/
│   └── useTranscriptions.ts      # Copia da option-X/hooks/
├── utils/
│   ├── formatters.ts
│   └── exportTranscript.ts
├── types/
│   └── index.ts
├── .env.local
├── package.json
└── tailwind.config.ts
```

---

## 🎯 Deployment

### Vercel (Consigliato)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Add environment variables in Vercel dashboard
# - NEXT_PUBLIC_SUPABASE_URL
# - NEXT_PUBLIC_SUPABASE_ANON_KEY
# - NEXT_PUBLIC_MODAL_WEBHOOK_URL

# Deploy to production
vercel --prod
```

---

## 🔧 Customization

### Cambio Costo per Minuto

`hooks/useTranscriptions.ts`:

```ts
const COST_PER_MINUTE = 0.01; // Modifica qui
```

### Cambio Scadenza File

`components/TranscriptionTable.tsx`:

```ts
expiresAt.setDate(expiresAt.getDate() + 30); // Cambia 30 con giorni desiderati
```

### Cambio Colori

`tailwind.config.ts`:

```ts
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          500: '#8b5cf6', // Viola (default)
          // Cambia con il tuo colore
        },
      },
    },
  },
};
```

---

## 🐛 Troubleshooting

### "Modal webhook failed"

- Verifica che Modal worker sia deployed: `modal app list`
- Controlla webhook URL in `.env.local`
- Verifica Modal secrets: `modal secret list`

### "Supabase RLS error"

- Verifica che utente sia autenticato
- Controlla policies RLS nel dashboard Supabase
- Verifica che `user_id` corrisponda a `auth.uid()`

### "File upload failed"

- Verifica bucket `audio-temp` esista
- Controlla RLS policies su storage
- Verifica file size < 500MB

### "Export non funziona"

- Verifica che `segments` sia presente nel database
- Controlla console browser per errori
- Verifica formattazione timestamps

---

## 📊 Monitoraggio Costi Modal

Modal addebita in base a:
- **GPU time**: ~$1.10/ora per A10G
- **Container time**: incluso

**Stima costi per 1 ora di audio:**

- Trascrizione: ~51 secondi (70x realtime) = ~$0.016
- **Totale: ~€0.015 per ora di audio**

**Consigli per ridurre costi:**

1. Usa `scaledown_window` appropriato (già configurato a 10 min)
2. Non tenere container sempre accesi
3. Batch processa file simili insieme
4. Monitora su Modal dashboard

---

## 🎓 Prossimi Passi

1. **Scegli la dashboard** che preferisci
2. **Setup Supabase** (database + storage)
3. **Copia i file** dalla cartella `option-X` al tuo progetto
4. **Configura environment variables**
5. **Test in locale**: `npm run dev`
6. **Deploy su Vercel**
7. **Test con file audio reali**

---

## 📞 Support

Per problemi o domande:
1. Controlla questa guida
2. Leggi README specifico della tua option
3. Verifica logs Modal: `modal app logs whisperx-transcription`
4. Controlla Supabase logs nel dashboard

---

## 🎉 Congratulazioni!

Hai tutto il necessario per lanciare la tua piattaforma di trascrizione audio professionale! 🚀

**Buon lavoro e buone trascrizioni!** 🎙️✨
