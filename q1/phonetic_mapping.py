"""
q1/phonetic_mapping.py
Phonetic mapping via Wav2Vec2 forced alignment (HuggingFace / torchaudio).

Pipeline:
  1. Load utterance from LibriSpeech.
  2. Run WAV2VEC2_ASR_BASE_960H acoustic model → per-frame emission logits.
  3. Run torchaudio.functional.forced_align (CTC) to get per-word frame spans.
  4. Compare word-boundary start times to cepstrum V/UV boundary start times.
  5. Report RMSE and plot overlay.

Model: facebook/wav2vec2-base-960h via torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
       Downloads ~360 MB on first run; cached in ~/.cache/torch/hub/checkpoints/.

API notes (torchaudio 2.x):
  - forced_align expects:
      log_probs : (1, T, C)  float  — batched 2-D emission, NOT squeezed
      targets   : (1, S)     int32  — batched flat token sequence
      input_lengths : (1,)   int32
      target_lengths: (1,)   int32
      blank     : int        — index 0 in WAV2VEC2_ASR_BASE_960H label set
  - The model MUST run on CPU for forced_align (no MPS kernel).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_librispeech
from voiced_unvoiced import extract_features, classify, boundaries

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F_audio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# forced_align has no MPS kernel — must stay on CPU
DEVICE   = torch.device("cpu")
BUNDLE   = WAV2VEC2_ASR_BASE_960H
LABELS   = BUNDLE.get_labels()   # ('-', '|', 'e', 't', 'a', 'o', ...)
BLANK    = 0                      # index 0 is the CTC blank '-'
CHAR2IDX = {c: i for i, c in enumerate(LABELS)}


# ── tokeniser ────────────────────────────────────────────────────────────────
def words_to_token_lists(words: list) -> list:
    """
    Convert each word (lowercase) to a list of label indices.
    Characters absent from the label vocabulary are silently dropped.
    WAV2VEC2_ASR_BASE_960H uses lowercase labels.
    """
    result = []
    for word in words:
        ids = [CHAR2IDX[c] for c in word.lower() if c in CHAR2IDX]
        if ids:
            result.append(ids)
    return result


# ── forced alignment ─────────────────────────────────────────────────────────
def forced_align(sig_np: np.ndarray, sr: int, words: list) -> list:
    """
    Run CTC forced alignment with WAV2VEC2_ASR_BASE_960H.
    Returns [(word, start_s, end_s), ...].
    """
    # ── load model (cached after first run) ──
    model    = BUNDLE.get_model().to(DEVICE)
    model_sr = BUNDLE.sample_rate          # 16 000 Hz

    # ── prepare waveform ──
    wav = torch.tensor(sig_np, dtype=torch.float32).unsqueeze(0)  # (1, T)
    if sr != model_sr:
        wav = torchaudio.functional.resample(wav, sr, model_sr)
    wav = wav.to(DEVICE)

    # ── acoustic emission  (1, T_frames, C) ──
    with torch.inference_mode():
        emission, _ = model(wav)

    em2d     = emission[0]           # (T_frames, C)
    T_frames = em2d.shape[0]
    # seconds per emission frame (Wav2Vec2 downsamples by stride ≈ 320 samples)
    spf = wav.shape[-1] / model_sr / T_frames

    # ── tokenise transcript ──
    tok_lists = words_to_token_lists(words)
    flat      = [t for tl in tok_lists for t in tl]
    if not flat:
        print("[Q1-PM] WARNING: no tokens produced — check label vocabulary")
        return []

    # ── forced_align requires batched (1, ...) tensors ──
    targets  = torch.tensor(flat, dtype=torch.int32).unsqueeze(0).to(DEVICE)  # (1, S)
    em_b     = em2d.unsqueeze(0)                                               # (1, T, C)
    in_len   = torch.tensor([T_frames],    dtype=torch.int32)
    tgt_len  = torch.tensor([len(flat)],   dtype=torch.int32)

    paths, _ = F_audio.forced_align(em_b, targets, in_len, tgt_len, blank=BLANK)
    path     = paths[0].tolist()   # length T_frames; each entry is a label index

    # ── merge consecutive identical non-blank runs into segments ──
    segments = []
    prev, s0 = BLANK, 0
    for fi, tok in enumerate(path):
        if tok != prev:
            if prev != BLANK:
                segments.append((prev, s0, fi - 1))
            s0, prev = fi, tok
    if prev != BLANK:
        segments.append((prev, s0, T_frames - 1))

    # ── greedily assign segments → words (left-to-right) ──
    word_bounds = []
    si = 0
    for word, tl in zip(words, tok_lists):
        if si >= len(segments) or not tl:
            continue
        wf = wl = None
        consumed = 0
        while consumed < len(tl) and si < len(segments):
            tv, fs, fe = segments[si]
            if tv in tl:
                wf = fs if wf is None else min(wf, fs)
                wl = fe
                consumed += 1
                si += 1
            else:
                break
        if wf is not None:
            word_bounds.append((word, float(wf * spf), float((wl + 1) * spf)))

    return word_bounds


# ── RMSE ──────────────────────────────────────────────────────────────────────
def rmse_boundaries(manual: list, model_b: list) -> float:
    """
    For each V/UV boundary start time, find the nearest word boundary start
    time, then return the root-mean-square of those minimum distances.
    """
    ms = np.array([b[0] for b in manual if b[2] in ("voiced", "unvoiced")])
    mb = np.array([b[1] for b in model_b])
    if not len(ms) or not len(mb):
        return float("nan")
    errs = [float(np.abs(mb - m).min()) for m in ms]
    return float(np.sqrt(np.mean(np.array(errs) ** 2)))


# ── plot ──────────────────────────────────────────────────────────────────────
def plot_alignment(sig, sr, manual, model_b, rmse_val, path):
    t   = np.arange(len(sig)) / sr
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, sig, "k", lw=0.3, alpha=0.7)

    COLS = {"silence": "lightgray", "unvoiced": "steelblue", "voiced": "coral"}
    for s, e, l in manual:
        ax.axvspan(s, e, alpha=0.18, color=COLS.get(l, "white"))

    ytop = float(np.abs(sig).max()) * 0.88
    for word, s, e in model_b:
        ax.axvline(s, color="green", lw=1.2, alpha=0.85)
        ax.text(s + 0.01, ytop, word[:6], fontsize=6.5,
                color="darkgreen", rotation=55, va="top")

    ax.set_xlim([0, t[-1]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"V/UV Boundaries (shaded) vs Wav2Vec2 CTC Word Boundaries (green)\n"
        f"Model: WAV2VEC2_ASR_BASE_960H  |  RMSE = {rmse_val * 1000:.1f} ms"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"[Q1-PM] plot saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[Q1-PM] label vocabulary ({len(LABELS)}): {LABELS[:10]} ...")

    ds = get_librispeech("test-clean")
    wav, sr, transcript, *_ = ds[0]

    # Use full utterance (no truncation) for better alignment quality
    sig   = wav.squeeze().numpy()
    words = transcript.split()
    print(f"[Q1-PM] transcript : {transcript}")
    print(f"[Q1-PM] words      : {words}")
    print(f"[Q1-PM] duration   : {len(sig)/sr:.2f}s")

    # ── cepstrum V/UV boundaries ──
    feats  = extract_features(sig, sr)
    lbl    = classify(feats)
    manual = boundaries(lbl, feats["hop"], sr)

    # ── Wav2Vec2 forced alignment (downloads model on first run) ──
    print("[Q1-PM] loading WAV2VEC2_ASR_BASE_960H and running forced alignment …")
    print("        (first run downloads ~360 MB to ~/.cache/torch/hub/checkpoints/)")
    model_b = forced_align(sig, sr, words)

    # ── RMSE ──
    rv = rmse_boundaries(manual, model_b)
    print(f"\n[Q1-PM] RMSE = {rv * 1000:.2f} ms")

    # ── word boundary table ──
    print(f"\n{'Word':<20} {'Start (s)':>10} {'End (s)':>10}")
    print("-" * 44)
    for w, s, e in model_b:
        print(f"{w:<20} {s:>10.3f} {e:>10.3f}")

    plot_alignment(sig, sr, manual, model_b, rv,
                   os.path.join(OUT, "phonetic_mapping.png"))
    print("\n[Q1-PM] phonetic_mapping.py completed successfully.")
