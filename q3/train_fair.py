"""
q3/train_fair.py
CTC-based ASR training with a custom FairnessLoss that minimises
the inter-group (gender proxy) CTC loss variance.

Fixes applied:
  1. MPS fix: aten::_ctc_loss has no MPS kernel.
              Model forward on MPS (fast); CTC loss computed on CPU.
  2. Convergence fix: reduced lr 1e-3→3e-4, fairness lambda 0.3→0.05,
              added ReduceLROnPlateau scheduler watching val CTC,
              increased max_frames 300→400 for better temporal coverage,
              added input LayerNorm before RNN for stable gradients.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_librispeech
from audit import gender_proxy

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torchaudio.transforms as T
import torchaudio.functional as FA
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "fair_results")
os.makedirs(RES, exist_ok=True)

SR    = 16000
CHARS = [" "] + list("abcdefghijklmnopqrstuvwxyz'")
BLANK = len(CHARS)
C2I   = {c: i for i, c in enumerate(CHARS)}

# Model forward pass on MPS when available; CTC loss always on CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CPU    = torch.device("cpu")
print(f"[Q3-Fair] model device = {DEVICE}  |  CTC loss device = cpu")


# ── helpers ───────────────────────────────────────────────────────────────────
def text2tns(t: str) -> torch.Tensor:
    return torch.tensor([C2I[c] for c in t.lower() if c in C2I], dtype=torch.long)


def pitch_fast(sig: np.ndarray, sr: int) -> float:
    """Fast autocorrelation-based F0 estimate on first 50 ms."""
    fl  = min(int(sr * 0.05), len(sig))
    f   = sig[:fl] - sig[:fl].mean()
    if f.std() < 1e-5:
        return 0.0
    acf = np.correlate(f, f, "full")[len(f):]
    lm, lx = int(sr / 400), int(sr / 60)
    if lx >= len(acf):
        return 0.0
    seg = acf[lm:lx]
    pk  = np.argmax(seg) + lm
    return float(sr / pk) if acf[pk] / (acf[0] + 1e-8) > 0.3 else 0.0


# ── dataset ───────────────────────────────────────────────────────────────────
class ASRDataset(Dataset):
    def __init__(self, max_samples: int = 400, max_frames: int = 400):
        ds_raw  = get_librispeech("test-clean", max_samples)
        mfcc_fn = T.MFCC(SR, 40,
                         melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 64})
        self.items = []
        for i in range(len(ds_raw)):
            wav, sr, transcript, *_ = ds_raw[i]
            if sr != SR:
                wav = FA.resample(wav, sr, SR)
            mfcc = mfcc_fn(wav).squeeze(0)        # (40, T)
            T_   = mfcc.shape[1]
            if T_ < max_frames:
                mfcc = nn.functional.pad(mfcc, (0, max_frames - T_))
            else:
                mfcc = mfcc[:, :max_frames]
            lbl   = text2tns(transcript)
            f0    = pitch_fast(wav.squeeze().numpy(), SR)
            gp    = gender_proxy(f0)
            g_lbl = {"male_proxy": 0, "female_proxy": 1}.get(gp, -1)
            self.items.append((mfcc, lbl, g_lbl))

    def __len__(self):        return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    mfccs, labels, genders = zip(*batch)
    mfccs  = torch.stack(mfccs)
    lens   = torch.tensor([len(l) for l in labels])
    padded = nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=BLANK)
    return mfccs, padded, lens, torch.tensor(genders)


# ── model ─────────────────────────────────────────────────────────────────────
class SmallASR(nn.Module):
    """
    Bidirectional GRU CTC-ASR.
    LayerNorm on input stabilises gradients and speeds convergence.
    """
    def __init__(self, idim: int = 40, hidden: int = 256, n_cls: int = None):
        super().__init__()
        if n_cls is None:
            n_cls = BLANK + 1
        self.norm = nn.LayerNorm(idim)           # ← stabilises training
        self.rnn  = nn.GRU(idim, hidden, num_layers=3,
                           batch_first=True, dropout=0.15,
                           bidirectional=True)
        self.fc   = nn.Linear(hidden * 2, n_cls)

    def forward(self, x):
        x = x.permute(0, 2, 1)        # (B, T, idim)
        x = self.norm(x)               # per-feature normalisation
        o, _ = self.rnn(x)
        return self.fc(o).log_softmax(-1)


# ── fairness loss ─────────────────────────────────────────────────────────────
class FairnessLoss(nn.Module):
    """
    Penalises variance of per-group mean CTC losses.
    L_fair = Var({ mean_CTC^(g) : g in groups })
    Encourages equal ASR performance across demographic groups.
    """
    def forward(self, per_sample: torch.Tensor,
                groups: torch.Tensor) -> torch.Tensor:
        gs = groups.unique()
        gs = gs[gs >= 0]                          # ignore 'unknown' group (-1)
        if len(gs) < 2:
            return torch.tensor(0., device=per_sample.device)
        means = [per_sample[groups == g].mean()
                 for g in gs if (groups == g).sum() > 0]
        if len(means) < 2:
            return torch.tensor(0., device=per_sample.device)
        return torch.stack(means).var()


# ── training loop ─────────────────────────────────────────────────────────────
def train(epochs: int = 20,
          lam:    float = 0.05,     # ← reduced from 0.3 to prevent instability
          max_samples: int = 400,
          batch:  int = 16,
          lr:     float = 3e-4):    # ← reduced from 1e-3 for stable convergence
    """
    Train the fairness-aware CTC-ASR model.

    Key hyperparameter rationale:
      lr = 3e-4  : Adam converges cleanly on CTC tasks at this rate;
                   1e-3 causes loss spikes on small datasets.
      lam = 0.05 : FairnessLoss is a variance of CTC values (~100-600);
                   even lam=0.3 adds thousands to the total loss, swamping
                   the CTC signal. 0.05 keeps the fairness gradient ~5% of
                   the total gradient norm.
      ReduceLROnPlateau: halves lr if val CTC stagnates for 3 epochs.
    """
    ds  = ASRDataset(max_samples)
    vn  = max(1, len(ds) // 10)
    tr, va = random_split(ds, [len(ds) - vn, vn])
    tr_ld  = DataLoader(tr, batch, shuffle=True,
                        collate_fn=collate, num_workers=0)
    va_ld  = DataLoader(va, batch, shuffle=False,
                        collate_fn=collate, num_workers=0)

    model = SmallASR().to(DEVICE)
    ctc   = nn.CTCLoss(blank=BLANK, reduction="none", zero_infinity=True)
    fl    = FairnessLoss()
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # ReduceLROnPlateau: halve lr if val CTC does not improve for 3 epochs
    sch   = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, verbose=True)

    hist  = {"ctc": [], "fair": [], "val": [], "lr": []}

    for ep in range(1, epochs + 1):
        model.train()
        ec = ef = nb = 0

        for mfcc, lbls, llens, genders in tr_ld:
            mfcc = mfcc.to(DEVICE)
            lp   = model(mfcc)                     # (B, T, C) on DEVICE

            # CTC loss must run on CPU (no MPS kernel)
            lpt       = lp.permute(1, 0, 2).to(CPU)          # (T, B, C)
            lbls_cpu  = lbls.to(CPU)
            llens_cpu = llens.to(CPU)
            ilen      = torch.full((lp.shape[0],), lp.shape[1],
                                   dtype=torch.long, device=CPU)

            ps          = ctc(lpt, lbls_cpu, ilen, llens_cpu)  # (B,)
            genders_cpu = genders.to(CPU)
            fair_term   = fl(ps, genders_cpu)
            loss        = ps.mean() + lam * fair_term

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            ec += ps.mean().item()
            ef += fair_term.item()
            nb += 1

        # ── validation ──
        model.eval()
        vc = 0
        with torch.no_grad():
            for mfcc, lbls, llens, _ in va_ld:
                mfcc      = mfcc.to(DEVICE)
                lp        = model(mfcc)
                lpt       = lp.permute(1, 0, 2).to(CPU)
                lbls_cpu  = lbls.to(CPU)
                llens_cpu = llens.to(CPU)
                ilen      = torch.full((lp.shape[0],), lp.shape[1],
                                       dtype=torch.long, device=CPU)
                vc += ctc(lpt, lbls_cpu, ilen, llens_cpu).mean().item()

        n_val   = max(len(va_ld), 1)
        avg_ctc = ec / nb
        avg_fair= ef / nb
        avg_val = vc / n_val
        cur_lr  = opt.param_groups[0]["lr"]

        sch.step(avg_val)   # adapt lr based on validation CTC

        hist["ctc"].append(avg_ctc)
        hist["fair"].append(avg_fair)
        hist["val"].append(avg_val)
        hist["lr"].append(cur_lr)

        print(f"[Q3-Fair] ep{ep:02d}/{epochs}  "
              f"ctc={avg_ctc:.2f}  fair={avg_fair:.2f}  "
              f"val={avg_val:.2f}  lr={cur_lr:.2e}")

    # ── save checkpoint + history ──
    ckpt_path = os.path.join(RES, "fair_asr.pt")
    torch.save(model.state_dict(), ckpt_path)
    with open(os.path.join(RES, "fair_hist.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"[Q3-Fair] checkpoint saved → {ckpt_path}")

    # ── plots ──
    ep_ax = range(1, len(hist["ctc"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ep_ax, hist["ctc"], label="Train CTC", marker="o", ms=4)
    axes[0].plot(ep_ax, hist["val"], label="Val CTC",   marker="s", ms=4)
    axes[0].set_title("CTC Loss (lower = better)")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep_ax, hist["fair"], color="red", marker="^", ms=4)
    axes[1].set_title("Fairness Loss (group CTC variance)")
    axes[1].set_xlabel("Epoch"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep_ax, hist["lr"], color="purple", marker=".", ms=4)
    axes[2].set_title("Learning Rate (ReduceLROnPlateau)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)

    plt.suptitle("Fairness-Aware CTC-ASR Training", fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(RES, "fair_curves.png")
    plt.savefig(fig_path, dpi=150)
    print(f"[Q3-Fair] curves saved → {fig_path}")
    print("[Q3-Fair] done.")
    return hist


if __name__ == "__main__":
    train(
        epochs=20,
        lam=0.05,          # fairness lambda: 0.05 keeps it ~5% of total gradient
        max_samples=400,
        batch=16,
        lr=3e-4,           # stable Adam lr for CTC on small datasets
    )
