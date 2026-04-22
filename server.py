"""
FastAPI server for VoxCPM2 TTS service.

Mirrors the chatterbox-turbo-tts API surface exactly so backends can swap
TTS providers without changing call-site code.

Key design:
- Reference audio stored once per voice (voices/<id>/ref.wav)
- Optional transcript stored for "ultimate" cloning mode (voices/<id>/transcript.txt)
- Warmup generation run on registration to prime VoxCPM2's internal cache
- Model weights loaded once at startup and kept on GPU
- True sentence-level streaming via model.generate_streaming()

Endpoints:
- GET  /health
- GET  /
- GET  /voices
- GET  /voice/{voice_id}
- POST /voice/register  — upload ref audio (+ optional transcript), warmup cache
- PATCH /voice/{voice_id} — rename
- DELETE /voice/{voice_id}
- POST /tts            — full WAV response
- POST /tts/stream     — raw PCM float32 chunked stream
- POST /tts/stream/wav — buffered WAV stream

Audio output: 24kHz mono PCM float32 (VoxCPM2 native)
"""

import asyncio
import ctypes
import io
import json
import logging
import os
import re
import shutil
import time
import uuid

# ---------------------------------------------------------------------------
# Fix: ensure libnvrtc-builtins.so.13.0 is findable by VoxCPM2's snake kernel.
# The cu13 library ships with the nvidia-cu13 Python package but isn't on the
# default LD_LIBRARY_PATH on RunPod images. Pre-load it so nvrtc can open it.
# ---------------------------------------------------------------------------
_NVRTC_BUILTINS_CANDIDATES = [
    "/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib",
    "/usr/local/lib/python3.11/dist-packages/nvidia/cuda_nvrtc/lib",
    "/usr/local/lib/ollama/mlx_cuda_v13",
]

def _fix_nvrtc_library_path():
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    additions = [p for p in _NVRTC_BUILTINS_CANDIDATES if os.path.isdir(p)]
    if additions:
        new_path = ":".join(additions + ([existing] if existing else []))
        os.environ["LD_LIBRARY_PATH"] = new_path
        # Also try to pre-load the library so the dynamic linker caches it
        for candidate_dir in additions:
            for name in ("libnvrtc-builtins.so.13.0", "libnvrtc-builtins.so"):
                lib_path = os.path.join(candidate_dir, name)
                if os.path.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
                    break

_fix_nvrtc_library_path()
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None  # VoxCPM instance — loaded once at startup
SAMPLE_RATE: int = 24000  # resolved from model after load; default matches VoxCPM2

VOICES_DIR = Path(os.getenv("VOICES_DIR", "./voices"))
# voices/<voice_id>/ref.wav          — reference audio for cloning
# voices/<voice_id>/transcript.txt   — (optional) transcript of ref.wav for ultimate-mode cloning
# voices/<voice_id>/meta.json        — {"name": "...", "created_at": "...", ...}

DEFAULT_CFG_VALUE = float(os.getenv("DEFAULT_CFG_VALUE", "2.0"))
DEFAULT_TIMESTEPS = int(os.getenv("DEFAULT_TIMESTEPS", "10"))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for sentence-level streaming."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Voice metadata helpers
# ---------------------------------------------------------------------------

def _read_voice_meta(voice_id: str) -> dict:
    meta_path = VOICES_DIR / voice_id / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"name": voice_id, "voice_id": voice_id}


def _write_voice_meta(voice_id: str, meta: dict):
    meta_path = VOICES_DIR / voice_id / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))


def _get_ref_wav_path(voice_id: str) -> Optional[str]:
    p = VOICES_DIR / voice_id / "ref.wav"
    return str(p) if p.exists() else None


def _get_transcript(voice_id: str) -> Optional[str]:
    p = VOICES_DIR / voice_id / "transcript.txt"
    return p.read_text().strip() if p.exists() else None


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _build_generate_kwargs(
    voice_id: Optional[str],
    cfg_value: float,
    inference_timesteps: int,
) -> dict:
    """
    Build keyword arguments for model.generate() / model.generate_streaming().

    If a voice is registered we pass its ref audio.
    If a transcript is also stored we use the "ultimate cloning" mode
    (prompt_wav_path + prompt_text), which gives higher voice similarity.
    """
    kwargs: dict = {
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
    }

    if not voice_id or voice_id == "default":
        return kwargs

    ref_wav = _get_ref_wav_path(voice_id)
    if ref_wav is None:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{voice_id}' not found. Register it via POST /voice/register",
        )

    transcript = _get_transcript(voice_id)

    kwargs["reference_wav_path"] = ref_wav

    if transcript:
        # Ultimate cloning: same clip used for both prompt and reference
        kwargs["prompt_wav_path"] = ref_wav
        kwargs["prompt_text"] = transcript

    return kwargs


def _do_warmup(voice_id: str, cfg_value: float, inference_timesteps: int):
    """
    Generate one short utterance with this voice to prime VoxCPM2's
    internal reference-audio processing cache.
    The first call with a new reference_wav_path triggers a ~10s compile;
    all subsequent calls are fast (~RTF 0.37).
    """
    warmup_text = "Hello, this is a warmup generation to initialise the voice cache."
    kwargs = _build_generate_kwargs(voice_id, cfg_value, inference_timesteps)
    logger.info(f"Warming up voice '{voice_id}'…")
    t0 = time.time()
    with torch.inference_mode():
        _ = model.generate(text=warmup_text, **kwargs)
    logger.info(f"Voice '{voice_id}' warmed up in {time.time() - t0:.2f}s")


def _generate_wav(text: str, generate_kwargs: dict) -> np.ndarray:
    """Synthesize text → float32 numpy array. Must be called inside inference_mode()."""
    wav = model.generate(text=text, **generate_kwargs)
    # VoxCPM2 returns a numpy array or torch tensor depending on version
    if isinstance(wav, torch.Tensor):
        wav = wav.squeeze().cpu().numpy()
    else:
        wav = np.asarray(wav, dtype=np.float32).squeeze()
    return wav.astype(np.float32)


def _to_wav_bytes(wav: np.ndarray) -> bytes:
    """Normalise + convert float32 array → WAV bytes."""
    max_val = np.abs(wav).max()
    if max_val > 0:
        wav = wav * (0.95 / max_val)
    wav = np.clip(wav, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down VoxCPM2 TTS server")


app = FastAPI(
    title="VoxCPM2 TTS Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "X-Audio-Format",
        "X-Audio-Encoding",
        "X-Audio-Sample-Rate",
        "X-Audio-Channels",
        "X-Generation-Time",
    ],
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model():
    global model, SAMPLE_RATE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"CUDA available: {torch.cuda.is_available()}, loading VoxCPM2 on {device}…")

    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

    # Resolve actual sample rate from model
    try:
        SAMPLE_RATE = model.tts_model.sample_rate
    except AttributeError:
        SAMPLE_RATE = 24000  # fallback
    logger.info(f"VoxCPM2 loaded. sample_rate={SAMPLE_RATE}")

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-warm all registered voices so first real call is fast
    count = 0
    for voice_dir in VOICES_DIR.iterdir():
        if voice_dir.is_dir() and (voice_dir / "ref.wav").exists():
            try:
                _do_warmup(voice_dir.name, DEFAULT_CFG_VALUE, DEFAULT_TIMESTEPS)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to pre-warm voice '{voice_dir.name}': {e}")
    if count:
        logger.info(f"Pre-warmed {count} registered voice(s)")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = Field(
        default=None,
        description="Registered voice ID for cloning. None = model default voice.",
    )
    cfg_value: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Classifier-free guidance strength. Higher = closer to reference voice.",
    )
    inference_timesteps: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Flow-matching timesteps. More = higher quality, slower.",
    )
    norm_loudness: bool = Field(
        default=True,
        description="Normalise output loudness.",
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducible output. None = random each time.",
    )


class VoiceUpdateRequest(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Endpoints — voice management
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": "VoxCPM2 (openbmb/VoxCPM2)",
        "sample_rate": SAMPLE_RATE,
        "registered_voices": _count_voices(),
        "streaming_supported": True,
    }


@app.get("/")
async def root():
    return {
        "service": "VoxCPM2 TTS Service",
        "version": "1.0.0",
        "model": "openbmb/VoxCPM2",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "registered_voices": _count_voices(),
        "endpoints": {
            "health": "GET /health",
            "voices": "GET /voices — list registered voices",
            "get_voice": "GET /voice/{voice_id}",
            "register_voice": "POST /voice/register — upload ref audio + optional transcript (one-time per voice)",
            "update_voice": "PATCH /voice/{voice_id} — rename",
            "delete_voice": "DELETE /voice/{voice_id}",
            "tts": "POST /tts — full WAV synthesis",
            "tts_stream": "POST /tts/stream — raw PCM float32 sentence-level streaming",
            "tts_stream_wav": "POST /tts/stream/wav — buffered WAV streaming",
        },
        "audio": {
            "format": "PCM float32 / WAV",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
        },
        "cloning_modes": {
            "basic": "Upload ref audio only — fast, good similarity",
            "ultimate": "Upload ref audio + transcript text — best similarity",
        },
    }


def _count_voices() -> int:
    if not VOICES_DIR.exists():
        return 0
    return sum(1 for d in VOICES_DIR.iterdir() if d.is_dir() and (d / "ref.wav").exists())


@app.get("/voices")
async def list_voices():
    voices = {}
    if VOICES_DIR.exists():
        for voice_dir in sorted(VOICES_DIR.iterdir()):
            if not voice_dir.is_dir():
                continue
            vid = voice_dir.name
            has_ref = (voice_dir / "ref.wav").exists()
            if not has_ref:
                continue
            meta = _read_voice_meta(vid)
            voices[vid] = {
                "name": meta.get("name", vid),
                "created_at": meta.get("created_at"),
                "duration_seconds": meta.get("duration_seconds"),
                "has_transcript": (voice_dir / "transcript.txt").exists(),
                "cloning_mode": "ultimate" if (voice_dir / "transcript.txt").exists() else "basic",
            }
    return {"voices": voices, "total": len(voices)}


@app.post("/voice/register")
async def register_voice(
    file: UploadFile = File(...),
    voice_id: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    transcript: Optional[str] = Form(
        None,
        description=(
            "Exact transcript of the reference audio. "
            "When provided, enables ultimate-cloning mode for higher voice similarity."
        ),
    ),
):
    """
    Register a new voice for cloning.

    Upload a reference audio clip (WAV/MP3/etc, >5 seconds clean speech).
    Optionally provide the exact transcript of that clip to unlock ultimate-cloning
    mode, which gives noticeably higher voice similarity at the cost of needing
    to transcribe the reference.

    This endpoint does the expensive work once:
    1. Saves ref.wav (and transcript.txt if provided)
    2. Runs a warmup generation with this voice to prime VoxCPM2's internal cache
    3. Every subsequent /tts call is fast (~RTF 0.37 on RTX 4090)
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    vid = voice_id or str(uuid.uuid4())[:8]
    vid = re.sub(r'[^a-zA-Z0-9_-]', '_', vid)

    voice_dir = VOICES_DIR / vid
    voice_dir.mkdir(parents=True, exist_ok=True)
    wav_dest = voice_dir / "ref.wav"

    # Save uploaded audio
    with open(wav_dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Validate duration
    try:
        import librosa
        audio, sr = librosa.load(str(wav_dest), sr=None)
        duration = len(audio) / sr
        if duration < 5.0:
            shutil.rmtree(voice_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio must be longer than 5 seconds (got {duration:.1f}s)",
            )
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(voice_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    # Save transcript if provided
    if transcript and transcript.strip():
        (voice_dir / "transcript.txt").write_text(transcript.strip())

    # Prime VoxCPM2's internal cache for this voice
    try:
        t0 = time.time()
        _do_warmup(vid, DEFAULT_CFG_VALUE, DEFAULT_TIMESTEPS)
        warmup_time = time.time() - t0
    except Exception as e:
        shutil.rmtree(voice_dir, ignore_errors=True)
        logger.error(f"Warmup failed for voice '{vid}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialise voice: {e}")

    # Write metadata
    voice_name = name or vid
    cloning_mode = "ultimate" if (transcript and transcript.strip()) else "basic"
    _write_voice_meta(vid, {
        "voice_id": vid,
        "name": voice_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 1),
        "cloning_mode": cloning_mode,
    })

    return {
        "voice_id": vid,
        "name": voice_name,
        "duration_seconds": round(duration, 1),
        "warmup_time_seconds": round(warmup_time, 2),
        "cloning_mode": cloning_mode,
        "message": (
            f"Voice '{voice_name}' ({vid}) registered in {cloning_mode} mode. "
            f"Use voice_id=\"{vid}\" in /tts requests."
        ),
    }


@app.get("/voice/{voice_id}")
async def get_voice(voice_id: str):
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists() or not (voice_dir / "ref.wav").exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    meta = _read_voice_meta(voice_id)
    return {
        **meta,
        "has_ref_audio": True,
        "has_transcript": (voice_dir / "transcript.txt").exists(),
        "cloning_mode": meta.get("cloning_mode", "basic"),
    }


@app.patch("/voice/{voice_id}")
async def update_voice(voice_id: str, request: VoiceUpdateRequest):
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    meta = _read_voice_meta(voice_id)
    meta["name"] = request.name
    _write_voice_meta(voice_id, meta)
    return {"voice_id": voice_id, "name": request.name, "message": "Voice updated"}


@app.delete("/voice/{voice_id}")
async def delete_voice(voice_id: str):
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    meta = _read_voice_meta(voice_id)
    shutil.rmtree(voice_dir, ignore_errors=True)
    return {"message": f"Voice '{meta.get('name', voice_id)}' ({voice_id}) deleted"}


# ---------------------------------------------------------------------------
# Endpoints — TTS synthesis
# ---------------------------------------------------------------------------

@app.post("/tts")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesise text to speech. Returns a complete WAV file.

    Use voice_id to select a registered voice for cloning.
    Omit voice_id to use VoxCPM2's built-in default voice.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    try:
        generate_kwargs = _build_generate_kwargs(
            request.voice_id, request.cfg_value, request.inference_timesteps
        )
    except HTTPException:
        raise

    try:
        logger.info(
            f"TTS: '{request.text[:80]}' voice={request.voice_id or 'default'} "
            f"cfg={request.cfg_value} steps={request.inference_timesteps}"
        )
        t0 = time.time()
        if request.seed is not None:
            torch.manual_seed(request.seed)
        with torch.inference_mode():
            wav = _generate_wav(request.text, generate_kwargs)

        wav_bytes = _to_wav_bytes(wav)
        elapsed = time.time() - t0
        logger.info(f"Generated {len(wav)/SAMPLE_RATE:.2f}s audio in {elapsed:.2f}s")

        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="tts_output.wav"',
                "X-Generation-Time": f"{elapsed:.3f}",
                "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}")


@app.post("/tts/stream")
async def synthesize_speech_stream(request: TTSRequest):
    """
    True streaming TTS using VoxCPM2's generate_streaming().

    Audio is delivered as raw PCM float32 chunks as each piece is generated.
    Typical first-chunk latency: 48-72ms on RTX 4090.

    Audio format: Raw PCM float32, mono, {SAMPLE_RATE}Hz
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    try:
        generate_kwargs = _build_generate_kwargs(
            request.voice_id, request.cfg_value, request.inference_timesteps
        )
    except HTTPException:
        raise

    async def audio_generator() -> AsyncGenerator[bytes, None]:
        try:
            logger.info(
                f"Stream: '{request.text[:80]}' voice={request.voice_id or 'default'}"
            )
            first_chunk = True
            t0 = time.time()

            if request.seed is not None:
                torch.manual_seed(request.seed)
            with torch.inference_mode():
                for chunk in model.generate_streaming(
                    text=request.text, **generate_kwargs
                ):
                    if first_chunk:
                        logger.info(f"First chunk in {(time.time()-t0)*1000:.0f}ms")
                        first_chunk = False

                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.squeeze().cpu().numpy()
                    chunk = np.asarray(chunk, dtype=np.float32).flatten()
                    chunk = np.clip(chunk, -1.0, 1.0)

                    yield chunk.tobytes()
                    await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            raise

    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Audio-Format": "pcm",
            "X-Audio-Encoding": "float32",
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
            "X-Audio-Channels": "1",
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache",
        },
    )


@app.post("/tts/stream/wav")
async def synthesize_speech_stream_wav(request: TTSRequest):
    """
    Stream TTS output as a complete WAV file delivered in chunks.

    Collects all streaming chunks, concatenates, normalises, then streams
    the WAV. Lower latency than /tts for long texts but higher than /tts/stream.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    try:
        generate_kwargs = _build_generate_kwargs(
            request.voice_id, request.cfg_value, request.inference_timesteps
        )
    except HTTPException:
        raise

    async def wav_generator() -> AsyncGenerator[bytes, None]:
        try:
            logger.info(
                f"WAV stream: '{request.text[:80]}' voice={request.voice_id or 'default'}"
            )
            all_chunks = []

            if request.seed is not None:
                torch.manual_seed(request.seed)
            with torch.inference_mode():
                for chunk in model.generate_streaming(
                    text=request.text, **generate_kwargs
                ):
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.squeeze().cpu().numpy()
                    all_chunks.append(np.asarray(chunk, dtype=np.float32).flatten())
                    await asyncio.sleep(0)

            if not all_chunks:
                return

            full_audio = np.concatenate(all_chunks)
            wav_bytes = _to_wav_bytes(full_audio)

            # Stream the WAV in 8KB chunks
            buf = io.BytesIO(wav_bytes)
            while True:
                data = buf.read(8192)
                if not data:
                    break
                yield data

        except Exception as e:
            logger.error(f"WAV streaming error: {e}", exc_info=True)
            raise

    return StreamingResponse(
        wav_generator(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="tts_output.wav"',
            "Transfer-Encoding": "chunked",
            "X-Audio-Sample-Rate": str(SAMPLE_RATE),
        },
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8009"))
    workers = int(os.getenv("WORKERS", "1"))  # keep at 1 — model is GPU-bound

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
