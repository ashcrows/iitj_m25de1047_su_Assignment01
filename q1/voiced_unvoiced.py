"""
q1/voiced_unvoiced.py
Cepstrum-based Voiced / Unvoiced / Silence boundary detection.
Low-quefrency  → vocal tract envelope
High-quefrency → pitch periodicity
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


def pre_emphasis(x, c=0.97):
    return np.append(x[0], x[1:] - c * x[:-1])

def hamming(N):
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2*np.pi*n/(N-1))

def next_pow2(n):
    p = 1
    while p < n: p <<= 1
    return p


def extract_features(sig: np.ndarray, sr: int,
                     frame_ms=25.0, hop_ms=10.0):
    fl  = int(sr * frame_ms / 1000)
    hl  = int(sr * hop_ms  / 1000)
    fft_sz = next_pow2(fl)
    win    = hamming(fl)
    sig    = pre_emphasis(sig)
    n_fr   = 1 + (len(sig) - fl) // hl

    q_ms  = np.arange(fft_sz//2 + 1) / sr * 1000
    low_m = q_ms < 1.5
    hi_m  = (q_ms >= 2.0) & (q_ms <= 16.0)

    zcr_l, en_l, lc_l, hc_l, f0_l = [], [], [], [], []

    for i in range(n_fr):
        s = i * hl
        f = sig[s: s+fl]
        if len(f) < fl: break
        f = f * win

        zcr_l.append(np.sum(np.abs(np.diff(np.sign(f)))) / (2*fl))
        en_l.append(np.log(np.sum(f**2) + 1e-10))

        spec = np.fft.rfft(f, n=fft_sz)
        cep  = np.fft.irfft(np.log(np.abs(spec)+1e-10), n=fft_sz)[:fft_sz//2+1]
        lc_l.append(np.mean(np.abs(cep[low_m])))
        hc_l.append(np.mean(np.abs(cep[hi_m])))

        # pitch via max in hi-quefrency cepstrum
        if hi_m.sum() > 0:
            pk_idx = np.argmax(np.abs(cep[hi_m]))
            q_bin  = np.where(hi_m)[0][pk_idx]
            q_s    = q_bin / sr
            f0_l.append(1.0 / q_s if q_s > 0 else 0.0)
        else:
            f0_l.append(0.0)

    return dict(zcr=np.array(zcr_l), energy=np.array(en_l),
                low_cep=np.array(lc_l), high_cep=np.array(hc_l),
                pitch=np.array(f0_l), hop=hl, fl=fl)


def classify(feats, en_pct=20, hc_pct=40):
    en = feats["energy"]; hc = feats["high_cep"]
    en_th = np.percentile(en, en_pct)
    hc_th = np.percentile(hc, hc_pct)
    lbl = np.zeros(len(en), dtype=int)
    act = en > en_th
    lbl[act & (hc <= hc_th)] = 1   # unvoiced
    lbl[act & (hc >  hc_th)] = 2   # voiced
    return lbl


def boundaries(lbl, hop, sr):
    names = {0:"silence", 1:"unvoiced", 2:"voiced"}
    segs, cur, s = [], lbl[0], 0
    for i in range(1, len(lbl)):
        if lbl[i] != cur:
            segs.append((s*hop/sr, i*hop/sr, names[cur]))
            s, cur = i, lbl[i]
    segs.append((s*hop/sr, len(lbl)*hop/sr, names[cur]))
    return segs


def plot(sig, sr, feats, lbl, segs, save_path):
    t   = np.arange(len(sig)) / sr
    ft  = np.arange(len(lbl)) * feats["hop"] / sr
    COLS = {"silence":"lightgray","unvoiced":"steelblue","voiced":"coral"}

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=False)

    ax = axes[0]
    ax.plot(t, sig, color="black", lw=0.4)
    for s, e, l in segs:
        ax.axvspan(s, e, alpha=0.25, color=COLS[l])
    handles = [plt.Rectangle((0,0),1,1,color=c,alpha=0.5) for c in COLS.values()]
    ax.legend(handles, list(COLS.keys()), loc="upper right", fontsize=8)
    ax.set_ylabel("Amplitude"); ax.set_title("Waveform + V/UV/Silence")
    ax.set_xlim([0, t[-1]])

    axes[1].plot(ft, feats["energy"], color="darkorange")
    axes[1].set_ylabel("Log Energy"); axes[1].set_xlim([0, t[-1]])

    axes[2].plot(ft, feats["high_cep"], color="steelblue")
    axes[2].set_ylabel("Hi-Cep"); axes[2].set_xlim([0, t[-1]])

    pitch = feats["pitch"].copy(); pitch[lbl!=2] = np.nan
    axes[3].plot(ft, pitch, color="green", lw=0.8)
    axes[3].set_ylabel("F0 (Hz)"); axes[3].set_xlabel("Time (s)")
    axes[3].set_ylim([0,600]); axes[3].set_xlim([0, t[-1]])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


if __name__ == "__main__":
    ds  = get_librispeech("test-clean")
    wav, sr, *_ = ds[0]
    sig = wav.squeeze().numpy()[:sr*10]

    feats = extract_features(sig, sr)
    lbl   = classify(feats)
    segs  = boundaries(lbl, feats["hop"], sr)

    v  = (lbl==2).mean()*100
    uv = (lbl==1).mean()*100
    si = (lbl==0).mean()*100
    print(f"[Q1-VUV] voiced={v:.1f}%  unvoiced={uv:.1f}%  silence={si:.1f}%  segs={len(segs)}")

    plot(sig, sr, feats, lbl, segs,
         os.path.join(OUT, "voiced_unvoiced.png"))
    print("[Q1-VUV] plot saved → outputs/voiced_unvoiced.png")
