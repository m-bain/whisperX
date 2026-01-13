# Option 1: Minimalist Professional Dashboard

Interfaccia pulita e professionale ispirata a Linear e Vercel.

## 🎨 Design Filosofia

"Less is more" - Focus su velocità, efficienza e chiarezza. Perfetta per power users che vogliono un'interfaccia rapida senza distrazioni.

## ✨ Features Implementate

### ✅ Tutte le funzionalità richieste

- **Supporto formati completo**: MP3, M4A, WAV, FLAC, OGG, WebM, AAC
- **Conteggio minuti utilizzati**: Visualizzato in real-time nel header
- **Tracking costi**: Calcolo automatico basato su minuti trascritti
- **Gestione scadenza file**: Mostra giorni rimanenti (30 giorni dalla creazione)
- **Status real-time con timer**:
  - `queued` → In coda
  - `processing` → Elaborazione (con timer live)
  - `completed` → Completata (mostra tempo totale)
  - `failed` → Fallita
- **Esportazione funzionante**: TXT, SRT, VTT, JSON con un click
- **Upload drag & drop**: Interfaccia intuitiva con validazione
- **Auto-refresh**: Aggiornamento automatico ogni 5s quando ci sono elaborazioni attive
- **Dark mode**: Design dark-first ottimizzato

### 📊 Dashboard Layout

```
┌─────────────────────────────────────────┐
│ Header (Sticky)                         │
├─────────────────────────────────────────┤
│ ┌──────┐ ┌──────┐ ┌──────┐             │
│ │ Min  │ │ Cost │ │Active│  Stats      │
│ └──────┘ └──────┘ └──────┘             │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────┐        │
│ │  Drag & Drop Upload Area    │        │
│ └─────────────────────────────┘        │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────┐        │
│ │  Transcriptions Table        │        │
│ │  (sortable, filterable)     │        │
│ └─────────────────────────────┘        │
└─────────────────────────────────────────┘
```

## 📁 Struttura File

```
option-1-minimalist/
├── components/
│   ├── Dashboard.tsx           # Main dashboard component
│   ├── StatsHeader.tsx         # Statistics cards (minutes, cost, active)
│   ├── UploadArea.tsx          # Drag & drop upload with validation
│   ├── TranscriptionTable.tsx  # Data table with sort/filter
│   ├── StatusBadge.tsx         # Status indicator with live timer
│   └── ExportDropdown.tsx      # Export menu (TXT, SRT, VTT, JSON)
├── hooks/
│   └── useTranscriptions.ts    # Supabase integration + real-time updates
├── utils/
│   ├── formatters.ts           # Date, duration, file size formatters
│   └── exportTranscript.ts     # Export logic for all formats
├── types/
│   └── index.ts                # TypeScript types
└── README.md                   # This file
```

## 🚀 Setup

### 1. Install Dependencies

```bash
npm install @supabase/supabase-js react-dropzone framer-motion
```

### 2. Environment Variables

Create `.env.local`:

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
NEXT_PUBLIC_MODAL_WEBHOOK_URL=your_modal_webhook_url
```

### 3. Supabase Database Schema

```sql
-- Transcripts table
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

-- Policy: Users can only see their own transcripts
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
```

### 4. Supabase Storage Buckets

Create two storage buckets:

1. **audio-temp** - For uploaded audio files
2. **transcripts** - For processed transcription JSON files

Both should have RLS policies allowing users to access only their own files.

### 5. Use in Next.js

```tsx
// app/dashboard/page.tsx
import { Dashboard } from '@/components/Dashboard';

export default function DashboardPage() {
  return <Dashboard />;
}
```

## 🎨 Customization

### Colors

Modifica `tailwind.config.js`:

```js
module.exports = {
  theme: {
    extend: {
      colors: {
        // Cambia il colore accent da violet a blue
        accent: {
          400: '#60a5fa',
          500: '#3b82f6',
        },
      },
    },
  },
};
```

### Cost Per Minute

Modifica in `hooks/useTranscriptions.ts`:

```ts
const COST_PER_MINUTE = 0.01; // €0.01 per minuto
```

### File Expiration

Modifica in `components/TranscriptionTable.tsx`:

```ts
expiresAt.setDate(expiresAt.getDate() + 30); // 30 giorni
```

## 🎯 Pro & Contro

### ✅ Pro

- **Veloce**: Minimal JS, caricamento rapido
- **Chiara**: Layout pulito senza distrazioni
- **Professionale**: Design sobrio e curato
- **Efficiente**: Ottima per uso quotidiano
- **Accessibile**: Contrasto e leggibilità ottimali

### ❌ Contro

- **Meno "wow"**: Design minimalista può sembrare semplice
- **Pochi grafici**: Focus su dati tabellari
- **Meno engagement**: Poche animazioni

## 📸 Screenshots

(Aggiungi qui screenshot della dashboard)

## 🔗 Next Steps

1. Aggiungi autenticazione Supabase
2. Deploy su Vercel
3. Configura Modal webhook
4. Test con file audio reali
5. Ottimizza performance

## 📝 Notes

- Questa dashboard usa Tailwind CSS per lo styling
- Richiede Next.js 14+ (App Router)
- Compatible con React 18+
- Ottimizzata per desktop (responsive mobile coming soon)
