"""
run_all.py  –  Master runner for Speech Assignment 1.
Runs Q1 → Q2 → Q3 in sequence. Dataset is downloaded once to data/.
"""

import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable


def run(label, script, cwd):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    r = subprocess.run([PY, script], cwd=cwd, check=True)
    print(f"  ✓ {label} complete")


if __name__ == "__main__":
    q1 = os.path.join(BASE, "q1")
    q2 = os.path.join(BASE, "q2")
    q3 = os.path.join(BASE, "q3")
    q3e = os.path.join(BASE, "q3", "evaluation_scripts")

    run("Q1 – Manual MFCC",            "mfcc_manual.py",       q1)
    run("Q1 – Leakage & SNR",           "leakage_snr.py",       q1)
    run("Q1 – Voiced/Unvoiced",         "voiced_unvoiced.py",   q1)
    run("Q1 – Phonetic Mapping",        "phonetic_mapping.py",  q1)

    run("Q2 – Train (Dis + Baseline)",  "train.py",             q2)
    run("Q2 – Evaluate",                "eval.py",              q2)

    run("Q3 – Bias Audit",              "audit.py",             q3)
    run("Q3 – Privacy Demo",            "pp_demo.py",           q3)
    run("Q3 – Fairness Training",       "train_fair.py",        q3)
    run("Q3 – FAD/DNSMOS Eval",         "fad_eval.py",          q3e)

    print("\n" + "="*60)
    print("  ALL QUESTIONS COMPLETE")
    print("="*60)
