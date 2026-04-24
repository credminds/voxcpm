"""
Dump VoxCPM internals to /tmp/voxcpm_inspect.txt so we know the exact
streaming API before editing server.py.

Run on the server:
    python inspect_voxcpm.py
    cat /tmp/voxcpm_inspect.txt

Then paste the file contents back.
"""
import inspect
import sys
import traceback

OUT = "/tmp/voxcpm_inspect.txt"
lines = []


def section(title):
    lines.append("")
    lines.append("=" * 72)
    lines.append(title)
    lines.append("=" * 72)


def safe(fn, label):
    try:
        fn()
    except Exception as e:
        lines.append(f"[ERROR in {label}] {type(e).__name__}: {e}")
        lines.append(traceback.format_exc())


# 1. Basic versions
section("1. Environment")
import voxcpm
lines.append(f"voxcpm location: {voxcpm.__file__}")
lines.append(f"voxcpm version: {getattr(voxcpm, '__version__', 'unknown')}")
try:
    import torch
    lines.append(f"torch: {torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")
except Exception as e:
    lines.append(f"torch import failed: {e}")


# 2. VoxCPM wrapper class — all public methods
section("2. VoxCPM wrapper class — public API")
from voxcpm.core import VoxCPM

public = [m for m in dir(VoxCPM) if not m.startswith("_")]
lines.append(f"public methods: {public}")

for name in ("generate", "generate_streaming"):
    if hasattr(VoxCPM, name):
        fn = getattr(VoxCPM, name)
        lines.append(f"\n--- VoxCPM.{name} signature ---")
        try:
            lines.append(str(inspect.signature(fn)))
        except Exception as e:
            lines.append(f"(signature unavailable: {e})")
        lines.append(f"\n--- VoxCPM.{name} source ---")
        try:
            lines.append(inspect.getsource(fn))
        except Exception as e:
            lines.append(f"(source unavailable: {e})")


# 3. Inner model — the real streaming generator
section("3. VoxCPM2Model (inner tts_model) — streaming internals")
from voxcpm.model.voxcpm2 import VoxCPM2Model

related = [m for m in dir(VoxCPM2Model)
           if ("generate" in m or "stream" in m or "chunk" in m) and not m.startswith("__")]
lines.append(f"streaming-related members: {related}")

for name in ("generate", "_generate", "generate_streaming"):
    if hasattr(VoxCPM2Model, name):
        fn = getattr(VoxCPM2Model, name)
        lines.append(f"\n--- VoxCPM2Model.{name} signature ---")
        try:
            lines.append(str(inspect.signature(fn)))
        except Exception as e:
            lines.append(f"(signature unavailable: {e})")


# 4. Full source of _generate — this is the actual streaming loop.
#    We want to see: what it yields, any chunk_size param, any min_chunk, etc.
section("4. VoxCPM2Model._generate — full source")
try:
    src = inspect.getsource(VoxCPM2Model._generate)
    lines.append(src)
except Exception as e:
    lines.append(f"(unavailable: {e})")


# 5. Full source of VoxCPM2Model.generate — the public method
section("5. VoxCPM2Model.generate — full source")
try:
    src = inspect.getsource(VoxCPM2Model.generate)
    lines.append(src)
except Exception as e:
    lines.append(f"(unavailable: {e})")


# 6. Sample-rate attribute
section("6. Sample rate")
lines.append(f"VoxCPM2Model has sample_rate on class? {hasattr(VoxCPM2Model, 'sample_rate')}")
# Look in __init__ for the sample_rate assignment
try:
    init_src = inspect.getsource(VoxCPM2Model.__init__)
    for line in init_src.splitlines():
        if "sample_rate" in line:
            lines.append(f"  init line: {line.strip()}")
except Exception as e:
    lines.append(f"(init source unavailable: {e})")


# 7. Chunk probe — what does one yielded chunk actually look like?
#    This requires loading the model, so it's gated behind a flag to avoid
#    wasting time if the user just wants signatures.
section("7. Live chunk probe (requires model load; skipped unless --probe)")
if "--probe" in sys.argv:
    lines.append("Loading VoxCPM for live probe...")
    try:
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
        sr = getattr(model.tts_model, "sample_rate", "unknown")
        lines.append(f"  model.tts_model.sample_rate = {sr}")

        import time
        t0 = time.time()
        first = None
        total_chunks = 0
        total_samples = 0
        for i, chunk in enumerate(model.generate_streaming(
            text="Hello world, this is a short test.",
            cfg_value=2.0,
            inference_timesteps=10,
        )):
            if first is None:
                first = time.time() - t0
                import numpy as np
                lines.append(f"  first chunk @ {first*1000:.0f}ms")
                lines.append(f"  type: {type(chunk).__name__}")
                if hasattr(chunk, "shape"):
                    lines.append(f"  shape: {chunk.shape}")
                if hasattr(chunk, "dtype"):
                    lines.append(f"  dtype: {chunk.dtype}")
                lines.append(f"  first-chunk duration (s): {len(chunk)/sr if isinstance(sr,(int,float)) else '?'}")
            total_chunks += 1
            total_samples += len(chunk) if hasattr(chunk, "__len__") else 0
        total = time.time() - t0
        lines.append(f"  total chunks: {total_chunks}")
        lines.append(f"  total audio samples: {total_samples}")
        if isinstance(sr, (int, float)):
            lines.append(f"  total audio duration: {total_samples/sr:.2f}s")
        lines.append(f"  total wall time: {total:.2f}s")
    except Exception as e:
        lines.append(f"[probe failed] {type(e).__name__}: {e}")
        lines.append(traceback.format_exc())
else:
    lines.append("(pass --probe to run a live model load + streaming test)")


with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"wrote {OUT} ({sum(len(l) for l in lines)} chars, {len(lines)} lines)")
print(f"run: cat {OUT}")
