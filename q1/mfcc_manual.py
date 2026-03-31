"""
q1/mfcc_manual.py
Manual MFCC / Cepstrum pipeline — no librosa.
Pre-emphasis → Windowing → FFT → Mel-Filterbank → Log → DCT
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_librispeech

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def pre_emphasis(x: np.ndarray, c: float = 0.97) -> np.ndarray:
    return np.append(x[0], x[1:] - c * x[:-1])

def make_frames(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    n = 1 + (len(x) - frame_len) // hop
    idx = (np.tile(np.arange(frame_len), (n, 1)) +
           np.tile(np.arange(0, n * hop, hop), (frame_len, 1)).T)
    return x[idx]

def window(N: int, kind: str) -> np.ndarray:
    n = np.arange(N)
    if kind == "hamming":   return 0.54 - 0.46 * np.cos(2*np.pi*n/(N-1))
    if kind == "hanning":   return 0.50 * (1 - np.cos(2*np.pi*n/(N-1)))
    return np.ones(N)

def hz2mel(f): return 2595.0 * np.log10(1 + f / 700.0)
def mel2hz(m): return 700.0 * (10**(m / 2595.0) - 1)

def mel_filterbank(n_filt: int, fft_sz: int, sr: int) -> np.ndarray:
    mels = np.linspace(hz2mel(0), hz2mel(sr/2), n_filt + 2)
    bins = np.floor((fft_sz+1) * np.array([mel2hz(m) for m in mels]) / sr).astype(int)
    fb = np.zeros((n_filt, fft_sz//2 + 1))
    for m in range(1, n_filt+1):
        lo, ctr, hi = bins[m-1], bins[m], bins[m+1]
        fb[m-1, lo:ctr] = (np.arange(lo, ctr) - lo) / max(ctr - lo, 1)
        fb[m-1, ctr:hi] = (hi - np.arange(ctr, hi)) / max(hi - ctr, 1)
    return fb

def dct2(x: np.ndarray, n_ceps: int) -> np.ndarray:
    N = x.shape[1]
    k = np.arange(n_ceps)[:, None]
    n = np.arange(N)
    D = np.cos(np.pi * k * (2*n + 1) / (2*N))
    return (x @ D.T)


def compute_mfcc(sig: np.ndarray, sr: int,
                 frame_ms=25, hop_ms=10, n_filt=26, n_ceps=13,
                 win="hamming") -> np.ndarray:
    fl = int(sr * frame_ms / 1000)
    hl = int(sr * hop_ms  / 1000)
    fft_sz = 1
    while fft_sz < fl: fft_sz <<= 1

    sig  = pre_emphasis(sig)
    frames = make_frames(sig, fl, hl)
    frames = frames * window(fl, win)

    power  = (1/fft_sz) * np.abs(np.fft.rfft(frames, n=fft_sz))**2
    fb     = mel_filterbank(n_filt, fft_sz, sr)
    log_e  = np.log(power @ fb.T + 1e-10)
    mfcc   = dct2(log_e, n_ceps)
    return mfcc                     # (T, n_ceps)


def compute_cepstrum(sig: np.ndarray, sr: int,
                     frame_ms=25, hop_ms=10, win="hamming") -> np.ndarray:
    fl = int(sr * frame_ms / 1000)
    hl = int(sr * hop_ms  / 1000)
    fft_sz = 1
    while fft_sz < fl: fft_sz <<= 1

    sig    = pre_emphasis(sig)
    frames = make_frames(sig, fl, hl) * window(fl, win)
    spec   = np.fft.rfft(frames, n=fft_sz)
    log_s  = np.log(np.abs(spec) + 1e-10)
    cep    = np.fft.irfft(log_s, n=fft_sz)[:, :fft_sz//2+1]
    return cep                      # (T, fft_sz//2+1)


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = get_librispeech("test-clean")
    wav, sr, transcript, *_ = ds[0]
    sig = wav.squeeze().numpy()[:sr*10]
    print(f"[Q1-MFCC] sr={sr}  len={len(sig)/sr:.1f}s")
    print(f"          transcript: {transcript[:60]}")

    mfcc = compute_mfcc(sig, sr)
    print(f"          MFCC shape: {mfcc.shape}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.linspace(0, len(sig)/sr, mfcc.shape[0])
    im = ax.imshow(mfcc.T, aspect="auto", origin="lower",
                   extent=[0, len(sig)/sr, 0, mfcc.shape[1]])
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("MFCC coefficient")
    ax.set_title("Manual MFCC (Hamming window, 13 coefficients)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "mfcc_manual.png"), dpi=150)
    print(f"[Q1-MFCC] plot saved → outputs/mfcc_manual.png")
