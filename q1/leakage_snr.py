"""
q1/leakage_snr.py
Spectral Leakage & SNR analysis for Rectangular, Hamming, Hanning windows.
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

WINDOWS = ["rectangular", "hamming", "hanning"]
COLORS  = {"rectangular": "steelblue", "hamming": "darkorange", "hanning": "green"}


def make_window(N: int, kind: str) -> np.ndarray:
    n = np.arange(N)
    if kind == "hamming":  return 0.54 - 0.46 * np.cos(2*np.pi*n/(N-1))
    if kind == "hanning":  return 0.50 * (1 - np.cos(2*np.pi*n/(N-1)))
    return np.ones(N)


def spectral_leakage(windowed: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(windowed))**2
    pk   = np.argmax(spec)
    hw   = max(2, len(spec)//16)
    lo, hi = max(0, pk-hw), min(len(spec), pk+hw+1)
    main = spec[lo:hi].sum()
    return float((spec.sum() - main) / (spec.sum() + 1e-30))


def snr_db(windowed: np.ndarray) -> float:
    spec   = np.abs(np.fft.rfft(windowed))**2
    sig_p  = np.mean(spec)
    noise  = np.percentile(spec, 5) + 1e-30
    return float(10 * np.log10(sig_p / noise))


def run(sig: np.ndarray, sr: int):
    # Pick a voiced frame at ~0.5 s
    fl  = int(sr * 0.025)
    start = min(int(0.5 * sr), len(sig) - fl)
    seg   = sig[start: start + fl]
    # zero-pad to next power-of-2
    fft_sz = 1
    while fft_sz < len(seg): fft_sz <<= 1
    seg = np.pad(seg, (0, fft_sz - len(seg)))
    N   = len(seg)

    results   = {}
    windowed  = {}
    for w in WINDOWS:
        ws = seg * make_window(N, w)
        windowed[w] = ws
        results[w]  = {"leakage": spectral_leakage(ws), "snr_db": snr_db(ws)}

    return results, windowed, seg, N, sr


def plot_and_save(results, windowed, seg, N, sr):
    freqs = np.fft.rfftfreq(N, 1/sr)

    # ── comparison plots ──
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    for row, w in enumerate(WINDOWS):
        ws  = windowed[w]
        win = make_window(N, w)

        ax_t = axes[row, 0]
        ax_t.plot(np.arange(N)/sr*1000, seg*win, color=COLORS[w], lw=0.8)
        ax_t.set_title(f"{w.capitalize()} – Windowed"); ax_t.set_xlabel("ms")
        ax_t.grid(True, alpha=0.3)

        spec_db = 20*np.log10(np.abs(np.fft.rfft(ws)) + 1e-10)
        ax_f = axes[row, 1]
        ax_f.plot(freqs, spec_db, color=COLORS[w], lw=0.8)
        ax_f.set_title(f"{w.capitalize()} – Log Spectrum")
        ax_f.set_xlabel("Hz"); ax_f.set_ylabel("dB"); ax_f.set_xlim([0, sr/2])
        ax_f.grid(True, alpha=0.3)

    plt.suptitle("Spectral Leakage Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(OUT, "leakage_comparison.png"), dpi=150)

    # ── bar chart ──
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    ws_list = list(WINDOWS); leak = [results[w]["leakage"] for w in ws_list]
    snr     = [results[w]["snr_db"] for w in ws_list]
    cols    = [COLORS[w] for w in ws_list]

    a1.bar(ws_list, leak, color=cols); a1.set_title("Leakage Ratio (↓ better)")
    a2.bar(ws_list, snr,  color=cols); a2.set_title("SNR dB (↑ better)")
    for ax in (a1, a2): ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "leakage_snr_bar.png"), dpi=150)

    return results


if __name__ == "__main__":
    ds = get_librispeech("test-clean")
    wav, sr, *_ = ds[0]
    sig = wav.squeeze().numpy()

    results, windowed, seg, N, sr = run(sig, sr)
    plot_and_save(results, windowed, seg, N, sr)

    print("\n" + "="*50)
    print(f"{'Window':<14} {'Leakage':>12} {'SNR (dB)':>10}")
    print("="*50)
    for w, v in results.items():
        print(f"{w:<14} {v['leakage']:>12.4f} {v['snr_db']:>10.2f}")
    print("="*50)
    print("[Q1-Leakage] plots saved → outputs/")
