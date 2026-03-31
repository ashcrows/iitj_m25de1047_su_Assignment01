"""
dataset.py  –  Shared LibriSpeech loader.
Scans project/data/LibriSpeech/<url>/ for .flac files.
Works with real LibriSpeech (via soundfile) and synthetic WAV-as-FLAC data.

Returned tuple mirrors torchaudio.datasets.LIBRISPEECH:
  (waveform_tensor, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
"""

import os, wave, glob
import numpy as np
import torch

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_ROOT, exist_ok=True)

_CACHE: dict = {}


class _LibriSpeechLocal:
    def __init__(self, root, url):
        base = os.path.join(root, "LibriSpeech", url)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Dataset not found: {base}")
        self._items = []
        for trans in sorted(glob.glob(os.path.join(base, "*", "*", "*.trans.txt"))):
            chapter_dir = os.path.dirname(trans)
            with open(trans) as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    parts  = line.split(" ", 1)
                    utt_id = parts[0]
                    text   = parts[1] if len(parts) > 1 else ""
                    segs   = utt_id.split("-")
                    spk    = segs[0]
                    chap   = segs[1] if len(segs) > 1 else "0"
                    apath  = os.path.join(chapter_dir, f"{utt_id}.flac")
                    if os.path.isfile(apath):
                        self._items.append((apath, spk, chap, utt_id, text))

    def __len__(self): return len(self._items)

    def __getitem__(self, i):
        apath, spk, chap, utt_id, text = self._items[i]
        wav, sr = _load(apath)
        return torch.tensor(wav).unsqueeze(0), sr, text, spk, chap, utt_id


def _load(path):
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        return (data[:,0] if data.ndim>1 else data), sr
    except Exception:
        pass
    with wave.open(path) as wf:
        sr  = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0, sr


class _Subset:
    def __init__(self, ds, n):
        self._ds = ds; self._n = min(n, len(ds))
    def __len__(self): return self._n
    def __getitem__(self, i):
        if i >= self._n: raise IndexError(i)
        return self._ds[i]


def get_librispeech(url="test-clean", max_samples=None):
    if url not in _CACHE:
        _CACHE[url] = _LibriSpeechLocal(DATA_ROOT, url)
    ds = _CACHE[url]
    return _Subset(ds, max_samples) if max_samples else ds
