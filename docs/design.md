# kokoro-tts: Design Spec v0.1.0

**Status:** Approved for implementation  
**Date:** 2026-10-16  
**Author:** Anton Feldmann  

---

## Executive Summary

Implementiere einen lokalen Text-to-Speech-Service unter `/Users/anton.feldmann/backup/Projects/comp/kokoro-tts` mit:
- FastAPI HTTP-API
- Typer CLI
- Kokoro 0.9.4 als Inferenz-Backend (PyTorch + Apple MPS support)
- WAV-Ausgabe (PCM 16-Bit, 22050 Hz, mono)

Keine Änderungen an bestehenden Projekten (`ai_creator_engine`).  
Alle Tests laufen ohne Modellgewichte (Fake-Backend).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        kokoro-tts                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐│
│  │   CLI (Typer) │    │  HTTP API    │     │ Fake Backend  ││
│  │  kokoro-tts  │────▶│ (FastAPI)    │◀───▶│  (for tests)  ││
│  └──────────────┘     └──────────────┘     └──────────────┘│
│                         │                                   │
│                    ┌────▼─────────────────────────┐         │
│                    │   SpeechService              │         │
│                    │   (shared logic)             │         │
│                    └─────────────────────────────┘         │
│                         │                                   │
│                    ┌────▼─────────────────────────┐         │
│                    │  InferenceBackend            │         │
│                    │  (Kokoro adapter)            │         │
│                    └─────────────────────────────┘         │
│                         │                                   │
│                    ┌────▼─────────────────────────┐         │
│                    │      Kokoro KPipeline        │         │
│                    │  (hexgrad/Kokoro-82M)        │         │
│                    └─────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

### Configuration (`config.py`)
- Pydantic Settings mit Umgebungsvariablen
- Model device resolution (auto → MPS/CPU)
- Cache directory management

### Schemas (`schemas.py`)
- `SpeechRequest`: input, language, voice, response_format
- `VoiceInfo`, `VoicesResponse`, `HealthResponse`

### Inference Adapter (`inference.py`)
- Kolooro KPipeline initialisation (einmalig beim Serverstart)
- Device management (MPS优先 auf Apple Silicon)
- Voice loading und management
- Phonemization via g2p/Misaki
- Text-to-Audio inference
- Thread-safe via RLock

### Speech Service (`service.py`)
- Request validation
- Voice/language selection
- Inferenz orchestration
- WAV encoding
- Exception mapping

### Audio Encoding (`audio.py`)
- WAV Header generation (RIFF/WAVE/fmt/data)
- 16-bit PCM conversion
- Sample rate preservation (22050 Hz von Kokoro)
- Deterministic output für Tests

### API (`api.py`)
- `GET /health`
- `GET /v1/audio/voices`
- `POST /v1/audio/speech`
- Error handling mit HTTP Statuscodes

### CLI (`cli.py`)
- `kokoro-tts serve`
- `kokoro-tts generate`
- `kokoro-tts voices`

### Exceptions (`exceptions.py`)
- `ModelNotLoadedError`
- `VoiceNotFoundError`
- `LanguageNotFoundError`
- `TextValidationError`
- `UnsupportedFormatError`
- `InferenceError`

---

## Initialization Strategy

**Decision:** Modell beim Serverstart laden (FastAPI lifespan).

```python
# In api.py or separate module:
@lifespan("startup")
async def on_startup():
    backend.load()  # Downloads model once
```

**Alternativ:** Server starten mit `503`, bis Modell geladen ist.  
**Chosen:** Starten und abbrechen bei Fehler, damit Status klar ist.

---

## Thread Safety

```python
class InferenceBackend:
    self._lock = threading.RLock()  # Reentrant lock
    
    def synthesize(self, text, language, voice):
        with self._lock:
            # Model inference here
            ...
```

RLock erlaubt gleiche Instanz mehrmals zu re-entrain (z.B. in service.py).

---

## Fake Backend Pattern

Für Tests ohne Modellgewichte:

```python
class FakeInferenceBackend:
    @property
    def model_loaded(self) -> bool:
        return False
    
    def synthesize(self, text, language, voice):
        # Generate fake audio (sine wave pattern)
        ...
```

Tests prüfen explizit `model_loaded` und trennen Fake- von echten Inferenz-Tests.

---

## Error Handling Matrix

| HTTP Code | Exception | Scenario |
|-----------|-----------|----------|
| 400 | TextValidationError | Empty/long text |
| 422 | Pydantic validation error | Invalid schema fields |
| 404 | VoiceNotFoundError | Unknown voice ID |
| 503 | ModelNotLoadedError | Model not loaded yet |
| 500 | InferenceError | Backend crash, unsupported format |

---

## Platform Compatibility

### Apple Silicon (M4 Max)
- PyTorch MPS: `torch.backends.mps.is_available()`
- Device auto-resolution: prefers MPS if available
- CPU fallback wenn MPS nicht verfügbar

### macOS Dependencies
- `espeakng-loader` ist Teil von `misaki[en]`
- System-wide `espeak-ng` Installation erforderlich (Homebrew):
  ```bash
  brew install espeak-ng
  ```

---

## Model Loading Behavior

```python
pipeline = kokoro.KPipeline(lang_code="en-us", repo_id="hexgrad/Kokoro-82M")
pipeline.load_voice("af_heart")  # Lazy loading
```

**Notes:**
- `list_voices()` gibt nur geladene Stimmen zurück
- Voice loading kann Downloads erfordern
- Fehler beim Laden werden als Exception geworfen

---

## Output Format

| Attribute | Value | Source |
|-----------|-------|--------|
| Sample Rate | 22050 Hz | Kokoro default |
| Channels | 1 (mono) | Model output |
| Bit Depth | 16-bit PCM | WAV encoding |
| Format | RIFF/WAVE/PCM | Standard WAV |

---

## Security Considerations

- Default bind: `127.0.0.1` (localhost only)
- 0.0.0.0 nur bei expliziter Option
- Keine externen APIs aufgerufen

---

## Next Steps

1. Implementiere fehlende Module (`api.py`, `cli.py`)
2. Schreibe Tests für alle Komponenten
3. Führe echte Inferenz mit geladenem Modell durch
4. Validiere WAV-Output (Header, Sample Rate, Dauer)
5. README.md mit Installation, CLI, Curl-Beispielen

---

**Review Request:** Bitte prüfe dieses Design und gib Feedback bevor ich mit der Implementierung fortfahre.
