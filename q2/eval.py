"""
q2/eval.py
Evaluate trained models: EER, cosine similarity, t-SNE plots.
Run AFTER train.py.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_librispeech
from train import (SpeakerDataset, DisentangledModel, BaselineModel,
                   CFG, DEVICE, RESULTS, CKPTS)

import numpy as np, torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve


def load_model(tag, disentangled, n_sp):
    if disentangled:
        m = DisentangledModel(CFG["n_mfcc"],CFG["emb_dim"],n_sp,CFG["num_envs"])
    else:
        m = BaselineModel(CFG["n_mfcc"],CFG["emb_dim"],n_sp)
    m.load_state_dict(torch.load(os.path.join(CKPTS,f"{tag}.pt"), map_location="cpu"))
    return m.to(DEVICE).eval()


@torch.no_grad()
def extract_emb(model, loader, disentangled):
    embs, spks = [], []
    for mfcc, spk_l, _ in loader:
        mfcc = mfcc.to(DEVICE)
        e = model(mfcc,0.)[2] if disentangled else model(mfcc)[1]
        embs.append(e.cpu().numpy()); spks.append(spk_l.numpy())
    return np.concatenate(embs), np.concatenate(spks)


def compute_eer(embs, spks, n=2000):
    rng  = np.random.default_rng(42)
    idx  = rng.integers(0, len(spks), (n,2))
    sims = []; truth = []
    for i,j in idx:
        a=embs[i]/(np.linalg.norm(embs[i])+1e-8)
        b=embs[j]/(np.linalg.norm(embs[j])+1e-8)
        sims.append(float(a@b))
        truth.append(1 if spks[i]==spks[j] else 0)
    fpr,tpr,_ = roc_curve(truth,sims)
    fnr=1-tpr
    i = np.argmin(np.abs(fpr-fnr))
    return float((fpr[i]+fnr[i])/2)


def plot_tsne(e_dis, s_dis, e_base, s_base, top=8):
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,(embs,spks,title) in zip(axes,[
            (e_dis, s_dis, "Disentangled"),
            (e_base,s_base,"Baseline")]):
        uniq,cnts = np.unique(spks,return_counts=True)
        keep = uniq[np.argsort(-cnts)[:top]]
        mask = np.isin(spks,keep)
        z = TSNE(2,random_state=42,perplexity=min(30,mask.sum()-1)).fit_transform(embs[mask])
        for k,s in enumerate(keep):
            idx=spks[mask]==s
            ax.scatter(z[idx,0],z[idx,1],s=10,label=f"Spk{s}",alpha=0.7)
        ax.set_title(f"{title} embeddings (t-SNE)")
        ax.legend(fontsize=6,markerscale=2); ax.grid(True,alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS,"tsne.png"),dpi=150)


def plot_eer_bar(eer_d, eer_b):
    fig,ax=plt.subplots(figsize=(5,4))
    ax.bar(["Disentangled","Baseline"],[eer_d,eer_b],color=["coral","steelblue"])
    ax.set_title("Speaker Verification EER (↓ better)")
    ax.set_ylabel("EER")
    for i,v in enumerate([eer_d,eer_b]):
        ax.text(i,v+0.002,f"{v:.4f}",ha="center")
    ax.grid(True,axis="y",alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS,"eer_bar.png"),dpi=150)


if __name__ == "__main__":
    ds     = SpeakerDataset()
    n_sp   = len(ds.spk_map)
    loader = DataLoader(ds, 64, shuffle=False, num_workers=0)

    m_dis  = load_model("disentangled", True,  n_sp)
    m_base = load_model("baseline",     False, n_sp)

    e_dis, s_dis   = extract_emb(m_dis,  loader, True)
    e_base, s_base = extract_emb(m_base, loader, False)

    eer_d = compute_eer(e_dis,  s_dis)
    eer_b = compute_eer(e_base, s_base)
    print(f"[Q2-Eval] Disentangled EER : {eer_d:.4f}")
    print(f"[Q2-Eval] Baseline EER     : {eer_b:.4f}")

    results = {"disentangled_eer": eer_d, "baseline_eer": eer_b}
    with open(os.path.join(RESULTS,"eval_results.json"),"w") as f:
        json.dump(results,f,indent=2)

    # load histories for table
    for tag in ["disentangled","baseline"]:
        p = os.path.join(RESULTS,f"{tag}_hist.json")
        if os.path.exists(p):
            h = json.load(open(p))
            print(f"[Q2-Eval] {tag} best val acc = {max(h['val_acc']):.4f}")

    plot_tsne(e_dis,s_dis,e_base,s_base)
    plot_eer_bar(eer_d, eer_b)
    print("[Q2-Eval] plots saved → results/")
