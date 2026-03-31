"""
q3/audit.py
Ethical / Documentation-Debt audit of LibriSpeech test-clean.
Bias proxies: pitch-based gender, speaking rate, SNR.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_librispeech

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

PLOTS = os.path.join(os.path.dirname(__file__), "audit_plots")
os.makedirs(PLOTS, exist_ok=True)


def pitch_acf(sig, sr, f_min=60, f_max=400):
    fl = min(int(sr*0.05), len(sig))
    f  = sig[:fl] - sig[:fl].mean()
    if f.std() < 1e-5: return 0.0
    acf = np.correlate(f, f, "full")[len(f):]
    lm, lx = int(sr/f_max), int(sr/f_min)
    if lx >= len(acf): return 0.0
    seg = acf[lm:lx]; pk = np.argmax(seg)+lm
    return float(sr/pk) if acf[pk]/(acf[0]+1e-8)>0.3 else 0.0

def gender_proxy(f0):
    if f0<=0: return "unknown"
    return "male_proxy" if f0<165 else "female_proxy"

def snr_db(sig):
    s = np.mean(sig**2)
    n = np.percentile(np.abs(sig),5)**2 + 1e-10
    return float(10*np.log10(s/n))

def speaking_rate(text, dur): return len(text.split())*1.5/max(dur,0.1)


def run_audit(max_samples=300):
    ds = get_librispeech("test-clean", max_samples)
    records = []
    stats   = defaultdict(list)
    for i in range(len(ds)):
        wav, sr, transcript, spk, *_ = ds[i]
        sig = wav.squeeze().numpy()
        dur = len(sig)/sr
        f0  = pitch_acf(sig, sr)
        gp  = gender_proxy(f0)
        rec = dict(spk=spk, dur=dur, f0=f0, gender=gp,
                   snr=snr_db(sig), rate=speaking_rate(transcript,dur))
        records.append(rec); stats[gp].append(rec)
        if (i+1)%50==0: print(f"  [{i+1}/{len(ds)}] audited")
    return records, dict(stats)


def print_report(records, stats):
    print("\n"+"="*58)
    print("   DOCUMENTATION DEBT AUDIT — LibriSpeech test-clean")
    print("="*58)
    total = len(records)
    for g, rs in sorted(stats.items()):
        pct  = len(rs)/total*100
        durs = np.mean([r["dur"] for r in rs])
        f0s  = np.mean([r["f0"] for r in rs if r["f0"]>0]) if any(r["f0"]>0 for r in rs) else 0
        snrs = np.mean([r["snr"] for r in rs])
        rts  = np.mean([r["rate"] for r in rs])
        spks = len({r["spk"] for r in rs})
        print(f"\n  {g}  ({len(rs)} utts / {pct:.1f}% | {spks} speakers)")
        print(f"    avg duration    : {durs:.2f}s")
        print(f"    avg F0          : {f0s:.1f} Hz")
        print(f"    avg SNR         : {snrs:.2f} dB")
        print(f"    avg speech rate : {rts:.2f} syl/s")
    grps = {g:len(r) for g,r in stats.items() if g!="unknown"}
    if grps:
        mx,mn = max(grps,key=grps.get), min(grps,key=grps.get)
        ratio = grps[mx]/max(grps[mn],1)
        print(f"\n  Imbalance ratio ({mx}/{mn}) = {ratio:.2f}x")
        if ratio>1.5: print("  ⚠ Significant imbalance detected.")
    print("="*58)


def plot_audit(records, stats, save_path):
    COLS={"male_proxy":"steelblue","female_proxy":"coral","unknown":"gray"}
    grps=[g for g in stats if g!="unknown"]
    fig,axes=plt.subplots(2,3,figsize=(14,8))

    # utterance count
    ax=axes[0,0]; cnts=[len(stats[g]) for g in grps]
    ax.bar(grps,cnts,color=[COLS[g] for g in grps])
    ax.set_title("Utterance count by gender proxy")
    for i,v in enumerate(cnts): ax.text(i,v+1,str(v),ha="center")

    # F0 distribution
    ax=axes[0,1]
    for g in grps:
        f0s=[r["f0"] for r in stats[g] if r["f0"]>0]
        ax.hist(f0s,bins=25,alpha=0.6,label=g,color=COLS[g])
    ax.axvline(165,color="red",ls="--",lw=1.5,label="threshold")
    ax.set_title("F0 distribution"); ax.legend(fontsize=7)

    # Duration
    ax=axes[0,2]
    for g in grps:
        ax.hist([r["dur"] for r in stats[g]],bins=20,alpha=0.6,label=g,color=COLS[g])
    ax.set_title("Duration (s)"); ax.legend(fontsize=7)

    # SNR boxplot
    ax=axes[1,0]
    bp=ax.boxplot([[r["snr"] for r in stats[g]] for g in grps],labels=grps,patch_artist=True)
    for p,g in zip(bp["boxes"],grps): p.set_facecolor(COLS[g])
    ax.set_title("SNR (dB) by group")

    # Speaking rate
    ax=axes[1,1]
    for g in grps:
        ax.hist([r["rate"] for r in stats[g]],bins=20,alpha=0.6,label=g,color=COLS[g])
    ax.set_title("Speaking rate (syl/s)"); ax.legend(fontsize=7)

    # Unique speakers
    ax=axes[1,2]
    spk_cnts=[len({r["spk"] for r in stats[g]}) for g in grps]
    ax.bar(grps,spk_cnts,color=[COLS[g] for g in grps])
    ax.set_title("Unique speakers")
    for i,v in enumerate(spk_cnts): ax.text(i,v+0.2,str(v),ha="center")

    plt.suptitle("LibriSpeech Bias Audit",fontsize=13,fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(save_path, dpi=150)
    print(f"[Q3-Audit] plot saved → {save_path}")


if __name__=="__main__":
    records, stats = run_audit(300)
    print_report(records, stats)
    plot_audit(records, stats, os.path.join(PLOTS,"audit_plots.png"))
    with open(os.path.join(PLOTS,"audit_records.json"),"w") as f:
        json.dump(records,f,indent=2,default=str)
    print("[Q3-Audit] done.")
