"""
q2/train.py
Disentangled Representation Learning for Speaker Recognition.
Architecture: TDNN speaker encoder + Gradient Reversal Layer (GRL)
              to disentangle environment from speaker identity.
Baseline: same TDNN without GRL.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_librispeech

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torchaudio.transforms as T
import torchaudio.functional as FA
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "results")
CKPTS   = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(RESULTS, exist_ok=True); os.makedirs(CKPTS, exist_ok=True)

# ── config ──────────────────────────────────────────────────────────────────
CFG = dict(
    url         = "test-clean",
    max_samples = 500,
    n_mfcc      = 40,
    sr          = 16000,
    max_frames  = 200,
    emb_dim     = 128,
    num_envs    = 3,
    epochs      = 20,
    batch       = 32,
    lr          = 1e-3,
    lambda_grl  = 0.5,
    seed        = 42,
)
torch.manual_seed(CFG["seed"])
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Q2] device = {DEVICE}")

# ── dataset ──────────────────────────────────────────────────────────────────
class SpeakerDataset(Dataset):
    def __init__(self):
        ds_raw = get_librispeech(CFG["url"], CFG["max_samples"])
        self.mfcc_fn = T.MFCC(CFG["sr"], CFG["n_mfcc"],
                               melkwargs={"n_fft":512,"hop_length":160,"n_mels":64})
        spk_ids = sorted({ds_raw[i][3] for i in range(len(ds_raw))})
        self.spk_map = {s: i for i, s in enumerate(spk_ids)}
        self.items = []
        for i in range(len(ds_raw)):
            wav, sr, _, spk, *_ = ds_raw[i]
            if sr != CFG["sr"]:
                wav = FA.resample(wav, sr, CFG["sr"])
            mfcc = self.mfcc_fn(wav).squeeze(0)        # (n_mfcc, T)
            T_  = mfcc.shape[1]
            if T_ < CFG["max_frames"]:
                mfcc = nn.functional.pad(mfcc,(0,CFG["max_frames"]-T_))
            else:
                mfcc = mfcc[:,:CFG["max_frames"]]
            spk_lbl = self.spk_map[spk]
            en = mfcc.pow(2).mean().item()
            env_lbl = 0 if en<1.0 else (1 if en<5.0 else 2)
            self.items.append((mfcc, spk_lbl, env_lbl))

    def __len__(self):  return len(self.items)
    def __getitem__(self, i): return self.items[i]

# ── GRL ──────────────────────────────────────────────────────────────────────
class _GRLFn(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()
    @staticmethod
    def backward(ctx, g):
        lam, = ctx.saved_tensors
        return -lam*g, None

def grl(x, lam=1.0): return _GRLFn.apply(x, lam)

# ── model blocks ─────────────────────────────────────────────────────────────
class TDNN(nn.Module):
    def __init__(self, i, o, k, d=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(i, o, k, dilation=d, padding=d*(k-1)//2),
            nn.BatchNorm1d(o), nn.ReLU())
    def forward(self, x): return self.net(x)

class SpeakerEncoder(nn.Module):
    def __init__(self, idim=40, edim=128):
        super().__init__()
        self.tdnn = nn.Sequential(
            TDNN(idim,256,5), TDNN(256,256,3,2), TDNN(256,256,3,3),
            TDNN(256,256,1), TDNN(256,1500,1))
        self.proj = nn.Sequential(nn.Linear(3000,edim), nn.BatchNorm1d(edim))
    def forward(self, x):
        h = self.tdnn(x)
        h = torch.cat([h.mean(-1), h.std(-1)], 1)
        return self.proj(h)

class SpeakerHead(nn.Module):
    def __init__(self, edim, n): super().__init__(); self.fc=nn.Linear(edim,n)
    def forward(self, x): return self.fc(x)

class EnvHead(nn.Module):
    def __init__(self, edim, n):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(edim,64),nn.ReLU(),nn.Linear(64,n))
    def forward(self, x, lam=1.0): return self.fc(grl(x, lam))

class DisentangledModel(nn.Module):
    def __init__(self, idim, edim, n_spk, n_env):
        super().__init__()
        self.enc  = SpeakerEncoder(idim, edim)
        self.spk  = SpeakerHead(edim, n_spk)
        self.env  = EnvHead(edim, n_env)
    def forward(self, x, lam=1.0):
        e = self.enc(x)
        return self.spk(e), self.env(e, lam), e

class BaselineModel(nn.Module):
    def __init__(self, idim, edim, n_spk):
        super().__init__()
        self.enc = SpeakerEncoder(idim, edim)
        self.spk = SpeakerHead(edim, n_spk)
    def forward(self, x):
        e = self.enc(x); return self.spk(e), e

# ── training loop ────────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, lam, disentangled):
    model.train(); ce=nn.CrossEntropyLoss()
    tot_spk=tot_env=tot_corr=tot=0
    for mfcc, spk_l, env_l in loader:
        mfcc=mfcc.to(DEVICE); spk_l=spk_l.to(DEVICE); env_l=env_l.to(DEVICE)
        opt.zero_grad()
        if disentangled:
            sp,ev,_ = model(mfcc, lam)
            loss = ce(sp,spk_l) + ce(ev,env_l)
            tot_env += ce(ev,env_l).item()
        else:
            sp,_ = model(mfcc)
            loss = ce(sp, spk_l)
        loss.backward(); opt.step()
        tot_spk  += ce(sp,spk_l).item()
        tot_corr += (sp.argmax(1)==spk_l).sum().item()
        tot      += len(spk_l)
    return tot_spk/len(loader), tot_env/len(loader), tot_corr/tot

@torch.no_grad()
def eval_epoch(model, loader, disentangled):
    model.eval(); tot_corr=tot=0
    for mfcc,spk_l,_ in loader:
        mfcc=mfcc.to(DEVICE); spk_l=spk_l.to(DEVICE)
        sp = model(mfcc,0.)[0] if disentangled else model(mfcc)[0]
        tot_corr += (sp.argmax(1)==spk_l).sum().item()
        tot      += len(spk_l)
    return tot_corr/tot


def run(disentangled: bool, tag: str):
    ds   = SpeakerDataset()
    n_sp = len(ds.spk_map)
    val_n = max(1, len(ds)//10)
    tr_ds, va_ds = random_split(ds, [len(ds)-val_n, val_n])
    tr_ld = DataLoader(tr_ds, CFG["batch"], shuffle=True,  num_workers=0)
    va_ld = DataLoader(va_ds, CFG["batch"], shuffle=False, num_workers=0)

    if disentangled:
        model = DisentangledModel(CFG["n_mfcc"],CFG["emb_dim"],n_sp,CFG["num_envs"]).to(DEVICE)
    else:
        model = BaselineModel(CFG["n_mfcc"],CFG["emb_dim"],n_sp).to(DEVICE)

    opt  = optim.Adam(model.parameters(), lr=CFG["lr"])
    sch  = optim.lr_scheduler.StepLR(opt, 5, 0.5)
    hist = {"train_acc":[],"val_acc":[],"spk_loss":[],"env_loss":[]}

    for ep in range(1, CFG["epochs"]+1):
        lam = CFG["lambda_grl"] * ep/CFG["epochs"]
        sl, el, ta = train_epoch(model, tr_ld, opt, lam, disentangled)
        va          = eval_epoch(model, va_ld, disentangled)
        sch.step()
        hist["train_acc"].append(ta); hist["val_acc"].append(va)
        hist["spk_loss"].append(sl);  hist["env_loss"].append(el)
        print(f"[{tag}] ep{ep:02d} spk_loss={sl:.4f} env_loss={el:.4f} "
              f"train_acc={ta:.3f} val_acc={va:.3f}")

    torch.save(model.state_dict(), os.path.join(CKPTS, f"{tag}.pt"))
    with open(os.path.join(RESULTS, f"{tag}_hist.json"),"w") as f:
        json.dump(hist, f, indent=2)
    return hist, model, n_sp


def plot_curves(h_dis, h_base):
    ep = range(1, len(h_dis["val_acc"])+1)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4))
    a1.plot(ep,h_dis["val_acc"],  label="Disentangled",marker="o")
    a1.plot(ep,h_base["val_acc"], label="Baseline",    marker="s")
    a1.set_title("Val Accuracy"); a1.legend(); a1.grid(True,alpha=0.3)
    a2.plot(ep,h_dis["spk_loss"],  label="Dis-spk",marker="o")
    a2.plot(ep,h_base["spk_loss"], label="Base-spk",marker="s")
    a2.set_title("Speaker CE Loss"); a2.legend(); a2.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS,"training_curves.png"), dpi=150)


if __name__ == "__main__":
    print("=== Disentangled Model ===")
    h_dis, _, n_sp = run(True,  "disentangled")
    print("\n=== Baseline Model ===")
    h_base, _, _   = run(False, "baseline")
    plot_curves(h_dis, h_base)
    print(f"\nDisentangled best val acc : {max(h_dis['val_acc']):.4f}")
    print(f"Baseline     best val acc : {max(h_base['val_acc']):.4f}")
    print("[Q2-Train] done. Checkpoints saved to checkpoints/")
