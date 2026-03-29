# iitj_m25de1047_su_Assignment01

## Code Guide

### Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **macOS note**: PyTorch will use MPS (Apple Silicon) automatically when available.  
> All scripts fall back to CPU if MPS is not detected.

---

### Dataset

All scripts auto-download **LibriSpeech `test-clean`** (~346 MB) on first run into `./data/`.  
For Q2 full training, edit `CONFIG["dataset_url"] = "train-clean-100"` in `q2/train.py`.

---

### Question 1 — Multi-Stage Cepstral Feature Extraction & Phoneme Boundary Detection

| Script | What it does |
|---|---|
| `q1/mfcc_manual.py` | Full manual MFCC pipeline (pre-emphasis → DCT) |
| `q1/leakage_snr.py` | Spectral leakage & SNR for Rectangular / Hamming / Hanning windows |
| `q1/voiced_unvoiced.py` | Cepstrum-based V/UV/Silence boundary detection |
| `q1/phonetic_mapping.py` | Wav2Vec2 forced alignment + RMSE vs manual boundaries |

```bash
cd q1
python mfcc_manual.py
python leakage_snr.py
python voiced_unvoiced.py
python phonetic_mapping.py    # downloads torchaudio MMS model on first run
```

**Outputs** saved to `q1/outputs/`:
- `mfcc_manual.png`
- `leakage_snr_comparison.png`, `leakage_snr_bar.png`
- `voiced_unvoiced_boundaries.png`
- `phonetic_mapping.png`

---

### Question 2 — Disentangled Representation Learning for Speaker Recognition

| Script | What it does |
|---|---|
| `q2/train.py` | Trains Disentangled model (TDNN + GRL) and Baseline TDNN |
| `q2/eval.py` | Computes EER, plots t-SNE of speaker embeddings |
| `q2/configs/default.yaml` | Hyperparameter reference |

```bash
cd q2
python train.py    # trains both models, saves checkpoints/
python eval.py     # loads checkpoints, prints EER, plots t-SNE
```

**Outputs** in `q2/results/`:
- `q2_training_curves.png`
- `embedding_tsne.png`
- `eer_comparison.png`
- `disentangled_history.json`, `baseline_history.json`, `eval_results.json`

**Checkpoints** in `q2/checkpoints/`:
- `disentangled_final.pt` — disentangled model weights
- `baseline_final.pt`     — baseline model weights

---

### Question 3 — Ethical Auditing & Privacy-Preserving Transformation

| Script | What it does |
|---|---|
| `q3/audit.py` | Programmatic bias audit (gender proxy, SNR, speaking rate) |
| `q3/privacymodule.py` | PyTorch PP module: pitch shift + tempo normalisation |
| `q3/pp_demo.py` | End-to-end demo: saves original + transformed audio pair |
| `q3/train_fair.py` | CTC ASR training with custom FairnessLoss |
| `q3/evaluation_scripts/fad_dnsmos_eval.py` | FAD proxy + DNSMOS proxy evaluation |

```bash
cd q3
python audit.py
python pp_demo.py
python privacymodule.py
python train_fair.py
python evaluation_scripts/fad_dnsmos_eval.py
```

**Outputs**:
- `q3/audit_plots/audit_plots.png`
- `q3/examples/demo_original.wav`, `demo_transformed.wav`
- `q3/examples/spectrogram_comparison.png`, `pp_demo_spectrograms.png`
- `q3/fair_results/fair_training_curves.png`
- `q3/evaluation_scripts/audio_quality_eval.png`, `eval_audio_quality.json`

---

### Project Structure

```
speech_assignment/
├── requirements.txt
├── README.md
├── data/                    # auto-created; LibriSpeech downloads here
├── q1/
│   ├── mfcc_manual.py
│   ├── leakage_snr.py
│   ├── voiced_unvoiced.py
│   ├── phonetic_mapping.py
│   └── outputs/
├── q2/
│   ├── train.py
│   ├── eval.py
│   ├── configs/
│   │   └── default.yaml
│   ├── results/
│   └── checkpoints/
└── q3/
    ├── audit.py
    ├── privacymodule.py
    ├── pp_demo.py
    ├── train_fair.py
    ├── evaluation_scripts/
    │   └── fad_dnsmos_eval.py
    ├── audit_plots/
    ├── examples/
    └── fair_results/
```

---

### Notes

- **Quick run**: All scripts default to `test-clean` (346 MB, ~2600 utterances, capped at 300–500 samples) so they complete in minutes on a laptop.
- **Full run**: Set `quick_run = False` and `dataset_url = "train-clean-100"` in Q2 `train.py` for production-quality results.
- **MPS**: PyTorch MPS backend is used on Apple Silicon automatically. Set `DEVICE = torch.device("cpu")` manually if you encounter MPS errors on older macOS versions.
