"""
q3/evaluation_scripts/fad_eval.py
FAD proxy (MFCC Fréchet distance) + DNSMOS proxy (SNR-based MOS) evaluation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_librispeech
from privacymodule import PrivacyPreservingModule

import json, numpy as np, torch, torchaudio
import torchaudio.transforms as T
from scipy.linalg import sqrtm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "evaluation_scripts")
os.makedirs(RES, exist_ok=True)
SR  = 16000
N_MFCC = 40

mfcc_fn = T.MFCC(SR, N_MFCC, melkwargs={"n_fft":512,"hop_length":160,"n_mels":64})


def mfcc_stats(wav):
    m = mfcc_fn(wav).squeeze(0).numpy().T   # (T, n_mfcc)
    mu  = m.mean(0)
    cov = np.cov(m, rowvar=False) + 1e-6*np.eye(N_MFCC)
    return mu, cov

def fad(mu1,c1,mu2,c2):
    d   = mu1-mu2
    sq  = sqrtm(c1@c2).real
    return float(max(0, d@d + np.trace(c1+c2-2*sq)))

def dnsmos_proxy(wav):
    w   = wav.squeeze().numpy()
    sig = np.mean(w**2)
    ns  = np.percentile(np.abs(w),5)**2+1e-10
    snr = 10*np.log10(sig/ns)
    return float(np.clip(1.0+snr/30*4, 1, 5))


def evaluate(n=50):
    ds = get_librispeech("test-clean", n)
    pp = PrivacyPreservingModule()

    mus_o, covs_o, mos_o = [], [], []
    mus_t, covs_t, mos_t = [], [], []

    for i in range(len(ds)):
        wav,sr,*_ = ds[i]
        if sr!=SR: wav=torchaudio.functional.resample(wav,sr,SR)
        tr = pp(wav)
        if tr.dim()==1: tr=tr.unsqueeze(0)
        mu_o,c_o = mfcc_stats(wav); mu_t,c_t = mfcc_stats(tr)
        mus_o.append(mu_o);  covs_o.append(c_o); mos_o.append(dnsmos_proxy(wav))
        mus_t.append(mu_t);  covs_t.append(c_t); mos_t.append(dnsmos_proxy(tr))
        if (i+1)%10==0: print(f"  [{i+1}/{len(ds)}]")

    fad_val = fad(np.mean(mus_o,0),np.mean(covs_o,0),
                  np.mean(mus_t,0),np.mean(covs_t,0))
    mo, mt  = np.mean(mos_o), np.mean(mos_t)

    print(f"\n[Q3-Eval] FAD proxy        : {fad_val:.4f}")
    print(f"[Q3-Eval] Orig  DNSMOS     : {mo:.3f}/5")
    print(f"[Q3-Eval] Trans DNSMOS     : {mt:.3f}/5")
    print(f"[Q3-Eval] MOS drop         : {mo-mt:.3f}")

    results={"fad":fad_val,"orig_mos":mo,"trans_mos":mt,"mos_drop":mo-mt}
    with open(os.path.join(RES,"eval_quality.json"),"w") as f:
        json.dump(results,f,indent=2)

    # bar chart
    fig,(a1,a2)=plt.subplots(1,2,figsize=(9,4))
    a1.bar(["Original","Transformed"],[mo,mt],color=["steelblue","coral"])
    a1.set_ylim([0,5.5]); a1.set_title("DNSMOS Proxy Score")
    for i,v in enumerate([mo,mt]): a1.text(i,v+0.05,f"{v:.3f}",ha="center")
    a2.bar(["FAD proxy"],[fad_val],color="darkorange")
    a2.set_title("FAD Proxy (↓ better)")
    a2.text(0,fad_val+0.05,f"{fad_val:.4f}",ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(RES,"quality_eval.png"),dpi=150)
    print(f"[Q3-Eval] plot saved → evaluation_scripts/quality_eval.png")
    return results


if __name__=="__main__":
    evaluate(50)
