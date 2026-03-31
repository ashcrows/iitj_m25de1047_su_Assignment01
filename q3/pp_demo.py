"""
q3/pp_demo.py
End-to-end privacy-preserving demo: saves original + transformed audio pair.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_librispeech
from privacymodule import PrivacyPreservingModule

import torch, torchaudio, torchaudio.transforms as T
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EX  = os.path.join(os.path.dirname(__file__), "examples")
os.makedirs(EX, exist_ok=True)
SR  = 16000


if __name__=="__main__":
    ds = get_librispeech("test-clean")
    wav, sr, transcript, spk, *_ = ds[0]
    if sr!=SR: wav=torchaudio.functional.resample(wav,sr,SR)

    print(f"[Q3-Demo] Speaker   : {spk}")
    print(f"[Q3-Demo] Transcript: {transcript[:60]}")

    pp  = PrivacyPreservingModule(semitones=4.0, rate=1.05, sr=SR)
    out = pp(wav)

    torchaudio.save(os.path.join(EX,"original.wav"),    wav, SR)
    torchaudio.save(os.path.join(EX,"transformed.wav"), out, SR)
    print(f"[Q3-Demo] Saved original.wav + transformed.wav → examples/")

    mel_fn = T.MelSpectrogram(SR, n_fft=1024, hop_length=256, n_mels=80)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,w,title in [(axes[0],wav,"Original"),(axes[1],out,"PP-Transformed (+4 st, 1.05x)")]:
        mel=mel_fn(w.squeeze(0).unsqueeze(0))
        mdb=10*torch.log10(mel+1e-9)
        im=ax.imshow(mdb.squeeze(0).numpy(),aspect="auto",origin="lower",
                     extent=[0,w.shape[-1]/SR,0,80])
        ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel("Mel bin")
        plt.colorbar(im,ax=ax)
    plt.suptitle("Mel Spectrogram: Privacy-Preserving Demo",fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(EX,"spectrogram_comparison.png"),dpi=150)
    print("[Q3-Demo] spectrogram_comparison.png saved.")
