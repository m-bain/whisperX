# 🔧 WhisperX MCP Server Setup Guide

## Cos'è MCP?

Il **Model Context Protocol (MCP)** di Anthropic permette a Claude Desktop di accedere a dati esterni come "context resources". Il server MCP WhisperX espone le tue trascrizioni direttamente in Claude Desktop!

## ✨ Features

- **📋 List Resources**: Vedi tutte le trascrizioni disponibili in Claude Desktop
- **📖 Read Resources**: Apri e leggi trascrizioni complete con timestamp e speaker labels
- **🔍 Search Tool**: Cerca keyword in tutte le trascrizioni
- **📊 Stats Tool**: Ottieni statistiche su lingua, durata, speaker
- **👥 Speakers Tool**: Elenca tutti gli speaker unici

---

## 🚀 Setup

### 1. Installa MCP SDK

```bash
pip install mcp supabase
```

### 2. Configura Environment Variables

Crea un file `.env` o esporta le variabili:

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your_service_role_key_here"
```

**Windows PowerShell**:
```powershell
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY="your_service_role_key_here"
```

### 3. Testa il Server

```bash
# Linux/Mac
python mcp_server.py

# Windows
python mcp_server.py
```

Se funziona, vedrai il server in ascolto (non stampa nulla, è normale).

---

## 📱 Integrazione con Claude Desktop

### Mac

1. Apri il file di configurazione:
   ```bash
   code ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

2. Aggiungi il server WhisperX:
   ```json
   {
     "mcpServers": {
       "whisperx": {
         "command": "python",
         "args": ["/absolute/path/to/whisperX/mcp_server.py"],
         "env": {
           "SUPABASE_URL": "https://your-project.supabase.co",
           "SUPABASE_SERVICE_ROLE_KEY": "your_service_role_key"
         }
       }
     }
   }
   ```

3. Riavvia Claude Desktop

### Windows

1. Apri il file di configurazione:
   ```powershell
   notepad "$env:APPDATA\Claude\claude_desktop_config.json"
   ```

2. Aggiungi il server WhisperX:
   ```json
   {
     "mcpServers": {
       "whisperx": {
         "command": "python",
         "args": ["C:\\Users\\YourName\\whisperX\\mcp_server.py"],
         "env": {
           "SUPABASE_URL": "https://your-project.supabase.co",
           "SUPABASE_SERVICE_ROLE_KEY": "your_service_role_key"
         }
       }
     }
   }
   ```

3. Riavvia Claude Desktop

---

## 🎯 Come Usare in Claude Desktop

### 1. Visualizza Trascrizioni Disponibili

Nella conversazione Claude Desktop, scrivi:

```
@whisperx
```

Apparirà un menu con tutte le trascrizioni disponibili!

### 2. Leggi una Trascrizione

Clicca su una trascrizione dal menu, oppure scrivi:

```
Leggimi la trascrizione di "audio_file.mp3"
```

Claude avrà accesso a:
- Full text con timestamps
- Speaker labels (se disponibili)
- Metadata (lingua, durata, data)

### 3. Cerca nelle Trascrizioni

```
Cerca "machine learning" in tutte le trascrizioni
```

Claude userà il tool `search_transcripts` automaticamente!

### 4. Analizza Statistiche

```
Dammi statistiche sulle mie trascrizioni
```

Claude userà il tool `get_transcript_stats`.

### 5. Trova Speaker

```
Elenca tutti gli speaker nelle mie trascrizioni
```

Claude userà il tool `get_speakers`.

---

## 🔧 Tools Disponibili

### `search_transcripts`

Cerca keyword o frasi in tutte le trascrizioni.

**Parametri**:
- `query` (required): Testo da cercare
- `language` (optional): Filtra per lingua (es. "it", "en")
- `min_duration` (optional): Durata minima in secondi

**Esempio**:
```json
{
  "query": "intelligenza artificiale",
  "language": "it",
  "min_duration": 60
}
```

### `get_transcript_stats`

Ottieni statistiche aggregate su tutte le trascrizioni.

**Parametri**:
- `include_failed` (optional): Includi trascrizioni fallite (default: false)

**Output**:
- Numero totale trascrizioni
- Durata totale e media
- Distribuzione lingue
- Distribuzione speaker

### `get_speakers`

Elenca tutti gli speaker unici con i file in cui appaiono.

**Output**:
- Lista speaker unici
- Count per speaker
- File associati

---

## 💡 Esempi d'Uso

### Esempio 1: Analisi Contenuti

```
Prompt in Claude Desktop:
"Analizza tutte le mie trascrizioni italiane e dimmi i temi principali discussi"

Claude:
1. Usa search_transcripts con language="it"
2. Legge le risorse trovate
3. Analizza e riassume i temi
```

### Esempio 2: Trova Citazioni

```
Prompt:
"Trova tutte le volte in cui qualcuno ha parlato di 'blockchain'"

Claude:
1. Usa search_transcripts con query="blockchain"
2. Mostra file e timestamp esatti
3. Può anche leggere il contesto completo
```

### Esempio 3: Report Multi-file

```
Prompt:
"Confronta le opinioni espresse in trascrizione1.mp3 e trascrizione2.mp3"

Claude:
1. Legge entrambe le risorse
2. Estrae opinioni da ciascuna
3. Genera confronto dettagliato
```

---

## 🐛 Troubleshooting

### "MCP server not responding"

- Verifica che `python mcp_server.py` funzioni standalone
- Controlla che le env vars SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY siano impostate
- Verifica path assoluto in `claude_desktop_config.json`

### "No resources found"

- Controlla che ci siano trascrizioni con status="completed" nel database
- Verifica connessione Supabase

### "Tool call failed"

- Controlla logs: `~/Library/Logs/Claude/mcp-server-whisperx.log` (Mac)
- Verifica che la query sia valida

---

## 🎉 Vantaggi MCP

✅ **Context Automatico**: Claude vede tutte le trascrizioni senza doverle copiare
✅ **Ricerca Intelligente**: Trova informazioni in secondi
✅ **Analisi Multi-file**: Confronta e aggrega dati da più trascrizioni
✅ **Sempre Aggiornato**: Nuove trascrizioni appaiono automaticamente
✅ **Privacy**: Dati restano sul tuo Supabase, nessun upload extra

---

## 🔐 Security Note

**IMPORTANTE**: Il SERVICE_ROLE_KEY ha accesso completo al database. Usalo solo in ambienti fidati (es. il tuo computer personale). NON condividere `claude_desktop_config.json` con altri.

Per ambienti condivisi, considera di:
1. Creare un utente Supabase read-only
2. Usare RLS (Row Level Security) policies
3. Limitare accesso con API key con scope limitato

---

## 📚 Documentazione MCP

- **MCP Protocol**: https://modelcontextprotocol.io
- **Anthropic MCP Docs**: https://docs.anthropic.com/claude/docs/mcp
- **MCP Python SDK**: https://github.com/anthropics/anthropic-mcp

---

Buon lavoro con il tuo MCP server WhisperX! 🚀
