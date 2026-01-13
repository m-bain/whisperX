# 🎉 CONFIGURAZIONE FUNZIONANTE MODAL WORKER - MILESTONE

**Data:** 18 Novembre 2024
**Branch:** `claude/project-status-review-01HkxUm3YMNY6vWEEhXVSH5U`
**Commit:** `3aa543e` - "fix: add matplotlib dependency for pyannote.audio"
**Tag Git:** `milestone-modal-working`

## ✅ STATUS: TRANSCRIPTION FUNZIONANTE!

Il Modal worker WhisperX è completamente operativo con:
- ✅ GPU NVIDIA A10G
- ✅ Trascrizione audio con WhisperX large-v3
- ✅ Speaker diarization con pyannote.audio
- ✅ Upload automatico risultati su Supabase
- ✅ CORS configurato per frontend Next.js

---

## 📦 CONFIGURAZIONE COMPLETA MODAL WORKER

### Immagine Docker - Stack Testato e Funzionante

```python
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

whisperx_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .run_commands("apt-get update")  # IMPORTANTE: Update package lists
    .apt_install(
        "git",
        "build-essential",      # C/C++ compiler (gcc, g++, make)
        "clang",                # Clang compiler (RICHIESTO da PyAV build)
        "pkg-config",           # RICHIESTO per building PyAV
        "ffmpeg",               # FFmpeg runtime
        "libavcodec-dev",       # FFmpeg development libraries
        "libavformat-dev",      # RICHIESTO per compilare PyAV from source
        "libavdevice-dev",      # Device handling library
        "libavfilter-dev",      # Audio/video filtering library
        "libavutil-dev",        # Utility library
        "libswscale-dev",       # Scaling library
        "libswresample-dev"     # Resampling library
    )
    # Installa wheel e setuptools PRIMA (dal PyPI standard)
    .pip_install("wheel", "setuptools")

    # Installa PyTorch 2.0.0 (versione testata da Modal)
    .pip_install(
        "torch==2.0.0",
        "torchaudio==2.0.0",
        "numpy<2.0",
        index_url="https://download.pytorch.org/whl/cu118",
    )

    # Installa PyAV separatamente PRIMA (per evitare errori di build)
    .pip_install("av==11.0.0")

    # Installa WhisperX 3.2.0 + dipendenze
    .pip_install(
        "git+https://github.com/m-bain/whisperx.git@v3.2.0",
        "ctranslate2==4.4.0",
        "matplotlib",  # RICHIESTO da pyannote.audio
        "supabase",
        "fastapi",
        "pydantic",
    )

    # CRITICAL: Force numpy 1.x DOPO WhisperX installation
    # (pyannote.audio cerca di installare numpy 2.x)
    .pip_install("numpy==1.26.4")
)
```

---

## 🔑 PUNTI CRITICI - NON MODIFICARE!

### 1. **CLANG Compiler**
```python
"clang",  # PyAV richiede SPECIFICAMENTE clang, non solo gcc!
```
❌ **ERRORE SENZA:** `error: command 'clang' failed: No such file or directory`

### 2. **Tutte e 7 FFmpeg Libraries**
```python
"libavcodec-dev",
"libavformat-dev",
"libavdevice-dev",    # ← NON omettere!
"libavfilter-dev",    # ← NON omettere!
"libavutil-dev",
"libswscale-dev",
"libswresample-dev"
```
❌ **ERRORE SENZA:** `Package libavdevice was not found in the pkg-config search path`

### 3. **NumPy 1.26.4 DOPO WhisperX**
```python
# Prima: installa WhisperX
.pip_install("git+https://github.com/m-bain/whisperx.git@v3.2.0", ...)

# Dopo: forza numpy 1.x (pyannote.audio installa numpy 2.x!)
.pip_install("numpy==1.26.4")
```
❌ **ERRORE SENZA:** `AttributeError: np.NaN was removed in the NumPy 2.0 release`

### 4. **Matplotlib per pyannote.audio**
```python
"matplotlib",  # pyannote.audio ne ha bisogno
```
❌ **ERRORE SENZA:** `ModuleNotFoundError: No module named 'matplotlib'`

### 5. **PyAV Installato Separatamente**
```python
# PRIMA: PyAV da solo
.pip_install("av==11.0.0")

# POI: WhisperX
.pip_install("git+https://github.com/m-bain/whisperx.git@v3.2.0")
```
❌ **ERRORE SENZA:** `ERROR: Failed building wheel for av`

### 6. **apt-get update OBBLIGATORIO**
```python
.run_commands("apt-get update")  # PRIMA di apt_install!
```
❌ **ERRORE SENZA:** Pacchetti non trovati

---

## 🌐 CORS Configuration (FastAPI)

```python
from fastapi.middleware.cors import CORSMiddleware

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

❌ **ERRORE SENZA:** `405 Method Not Allowed` su OPTIONS request

---

## 📊 Stack Versions - Testato e Funzionante

| Componente | Versione | Note |
|------------|----------|------|
| **CUDA** | 12.4.0-devel | Full CUDA toolkit |
| **Python** | 3.11 | |
| **PyTorch** | 2.0.0 + cu118 | CUDA 11.8 compatible |
| **torchaudio** | 2.0.0 | |
| **NumPy** | 1.26.4 | ULTIMA versione 1.x stabile |
| **PyAV** | 11.0.0 | FFmpeg bindings |
| **WhisperX** | 3.2.0 | Dal repository GitHub |
| **ctranslate2** | 4.4.0 | Fast transformer inference |
| **matplotlib** | latest | Per pyannote.audio |
| **Supabase** | latest | Database client |
| **FastAPI** | latest | Web framework |

---

## 🚀 Deploy Instructions

```bash
# 1. Autenticazione Modal (solo prima volta)
modal token new

# 2. Deploy del worker
modal deploy modal_worker.py

# 3. Verifica deployment
# Vai su https://modal.com/apps e controlla lo status
```

---

## 🔧 Troubleshooting

### Se NumPy torna a 2.x
**Causa:** Qualche dipendenza lo reinstalla
**Fix:** Aggiungi un'altra riga `.pip_install("numpy==1.26.4")` alla fine

### Se PyAV non compila
**Causa:** Manca clang o FFmpeg libraries
**Fix:** Verifica che TUTTI i 7 pacchetti libav* siano installati

### Se CORS non funziona
**Causa:** Middleware non configurato
**Fix:** Verifica che CORSMiddleware sia PRIMA dei route handlers

---

## 📝 Cronologia Errori Risolti

1. ❌ `pkg-config not found` → ✅ Aggiunto pkg-config
2. ❌ `libavdevice not found` → ✅ Aggiunte tutte e 7 FFmpeg libs
3. ❌ `command 'clang' failed` → ✅ Aggiunto clang compiler
4. ❌ `NumPy 2.x incompatible` → ✅ Forzato numpy==1.26.4 dopo WhisperX
5. ❌ `matplotlib not found` → ✅ Aggiunto matplotlib
6. ❌ `405 Method Not Allowed` → ✅ Aggiunto CORS middleware

---

## ⚠️ NON FARE MAI

- ❌ NON rimuovere `clang` (PyAV lo richiede specificamente)
- ❌ NON rimuovere nessuna delle 7 FFmpeg libraries
- ❌ NON installare PyAV insieme a WhisperX (installare separatamente)
- ❌ NON omettere `apt-get update`
- ❌ NON usare numpy>=2.0 (incompatibile con PyTorch 2.0.0)
- ❌ NON cambiare l'ordine di installazione dei pacchetti

---

## 🎯 QUESTA CONFIGURAZIONE FUNZIONA!

**Se hai bisogno di tornare a questo punto:**

```bash
# Checkout del tag milestone
git checkout milestone-modal-working

# Oppure checkout del commit specifico
git checkout 3aa543e

# Deploy
modal deploy modal_worker.py
```

---

## 📧 Secrets Richiesti su Modal

```bash
modal secret create supabase-credentials \
  SUPABASE_URL=your_url \
  SUPABASE_SERVICE_ROLE_KEY=your_key
```

---

**FINE DOCUMENTAZIONE MILESTONE** 🎉
