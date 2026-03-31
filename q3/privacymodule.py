"""
q3/privacymodule.py
Privacy-Preserving voice attribute transformation.
Pitch shift via STFT magnitude interpolation + Griffin-Lim resynthesis.
"""

import torch, torch.nn as nn
import torchaudio, torchaudio.transforms as T
import torchaudio.functional as FA
import numpy as np


class PitchShifter(nn.Module):
    """Shift pitch by semitones via STFT frequency-axis interpolation."""
    def __init__(self, semitones=4.0, n_fft=1024, hop=256):
        super().__init__()
        self.ratio = 2**(semitones/12)
        self.spec  = T.Spectrogram(n_fft=n_fft, hop_length=hop, power=None)
        self.gl    = T.GriffinLim(n_fft=n_fft, hop_length=hop, n_iter=60, power=1.0)

    def forward(self, wav):           # wav: (1,T)
        mag = self.spec(wav).abs()    # (1,F,t)
        F   = mag.shape[1]
        out = torch.zeros_like(mag)
        for fi in range(F):
            src = fi / self.ratio
            lo  = int(src); hi = lo+1; frac = src-lo
            if 0<=lo<F and hi<F:
                out[0,fi] = (1-frac)*mag[0,lo] + frac*mag[0,hi]
            elif 0<=lo<F:
                out[0,fi] = mag[0,lo]
        wav_out = self.gl(out)
        if wav_out.dim()==1: wav_out=wav_out.unsqueeze(0)
        return wav_out


class TempoScaler(nn.Module):
    """Change speaking rate via resampling trick."""
    def __init__(self, rate=1.05, sr=16000):
        super().__init__()
        self.new_sr = int(sr*rate); self.sr=sr
    def forward(self, wav):
        return FA.resample(wav, self.new_sr, self.sr)


class PrivacyPreservingModule(nn.Module):
    """Male-old → Female-young: raise pitch, slightly faster tempo."""
    def __init__(self, semitones=4.0, rate=1.05, sr=16000):
        super().__init__()
        self.shift = PitchShifter(semitones)
        self.tempo = TempoScaler(rate, sr)

    @torch.no_grad()
    def forward(self, wav):   # wav: (1,T)
        x = self.shift(wav)
        x = self.tempo(x)
        if x.dim()==1: x=x.unsqueeze(0)
        if x.dim()==3: x=x.squeeze(0)
        return x
