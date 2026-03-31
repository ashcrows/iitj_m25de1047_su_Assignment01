# Speech Understanding — Assignment 1
**IITJ M25DE1047**

## Quick Start (macOS)

```bash
# 1. Prerequisites
brew install ffmpeg python@3.11

# 2. Create virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run all questions (dataset downloads automatically to data/ on first run)
python run_all.py

# 5. Generate PDF reports
python generate_reports.py
```

---

## Project Structure

```
project/
├── dataset.py              ← Shared LibriSpeech loader (used by Q1, Q2, Q3)
├── run_all.py              ← Master runner (runs all scripts in order)
├── generate_reports.py     ← Builds all 3 PDF reports
├── requirements.txt
├── README.md
│
├── data/                   ← Dataset lives here (auto-created)
│   └── LibriSpeech/test-clean/   ← downloaded once, reused by all questions
│
├── q1/
│   ├── mfcc_manual.py      ← Manual MFCC pipeline (no librosa)
│   ├── leakage_snr.py      ← Spectral leakage & SNR for 3 window types
│   ├── voiced_unvoiced.py  ← Cepstrum-based V/UV/Silence boundary detection
│   ├── phonetic_mapping.py ← Word boundary alignment + RMSE
│   └── outputs/            ← Plots saved here
│
├── q1_report.pdf           ← Generated report
│
├── q2/
│   ├── train.py            ← Train disentangled + baseline speaker models
│   ├── eval.py             ← EER, t-SNE, comparison plots
│   ├── configs/default.yaml
│   ├── results/            ← Plots + JSON metrics
│   ├── checkpoints/        ← Saved model weights (.pt)
│   └── review.pdf          ← Generated report + paper review
│
└── q3/
    ├── audit.py                        ← Bias audit of LibriSpeech
    ├── privacymodule.py                ← PyTorch PP voice transformer
    ├── pp_demo.py                      ← Demo: saves original + transformed WAV
    ├── train_fair.py                   ← Fairness-loss CTC ASR training
    ├── evaluation_scripts/fad_eval.py  ← FAD proxy + DNSMOS proxy
    ├── audit_plots/                    ← Audit visualisations
    ├── examples/                       ← original.wav + transformed.wav
    ├── fair_results/                   ← Fairness training curves
    └── q3_report.pdf                   ← Generated report
```

---

## Dataset

All three questions share **one copy** of LibriSpeech `test-clean` stored in `data/`.

- `dataset.py` is the single entry point — import `get_librispeech()` from any script.
- The dataset is downloaded **once** on first use and never re-downloaded.
- For full-scale Q2 training, change `url = "train-clean-100"` in `q2/configs/default.yaml`.

---

## Running Individual Questions

```bash
# Q1
cd q1
python mfcc_manual.py
python leakage_snr.py
python voiced_unvoiced.py
python phonetic_mapping.py

# Q2
cd q2
python train.py    # trains both models (~5 min on MPS)
python eval.py     # loads checkpoints, computes EER, plots t-SNE

# Q3
cd q3
python audit.py
python pp_demo.py
python train_fair.py
python evaluation_scripts/fad_eval.py
```

---

## Device Support

Scripts auto-detect **Apple Silicon MPS** and fall back to CPU.
Set `DEVICE = torch.device("cpu")` manually to override.

---

## Key Results Summary

| Question | Key Result |
|---|---|
| Q1 MFCC | 13-coefficient manual pipeline, shape (298, 13) |
| Q1 Leakage | Hamming best SNR: 28.91 dB; lowest leakage: Hanning 0.0201 |
| Q1 V/UV | Voiced 46%, Unvoiced 34%, Silence 20% |
| Q1 RMSE | 105.63 ms boundary alignment error |
| Q2 Disentangled | Best val acc 1.000, EER 0.0000 |
| Q2 Baseline | Best val acc 0.000, EER 0.0000 |
| Q3 Audit | 2.0× male/female-proxy imbalance detected |
| Q3 DNSMOS | Orig 3.642 / Trans 4.031 (no degradation) |
| Q3 FAD | 4482.99 (pitch shift causes distribution shift) |
