"""
step5_train/champion_challenger.py
Champion / Challenger Model Evaluation for VulnDetector AutoML Loop
Authors: Josiah Chuku & Dr. Jinwei Liu, FAMU 2026

NEW CONTRIBUTION vs capstone:
- Automated model promotion decision based on statistical significance
- Prevents regression: challenger must exceed champion by >1% F1 AND p<0.05
- Saves promotion decision + full metrics to JSON for Azure DevOps gate
"""

import argparse
import json
import os
import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    f1_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score,
)

# Import your existing model and dataset classes
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step4_model.full_model import VulnDetector
from step5_train.prepare_data import VulnDataset


def load_model(checkpoint_path, device):
    """Load a VulnDetector checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = VulnDetector()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def get_predictions(model, dataloader, device, threshold=0.7341):
    """Run inference and return scores + binary predictions."""
    all_scores, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k != "label"}
            labels = batch["label"].numpy()
            logits = model(**inputs)
            scores = torch.sigmoid(logits).cpu().numpy()
            preds  = (scores >= threshold).astype(int)
            all_scores.extend(scores.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_scores), np.array(all_preds), np.array(all_labels)


def compute_metrics(scores, preds, labels):
    """Compute full metric suite."""
    return {
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "auc":       float(roc_auc_score(labels, scores)),
        "mcc":       float(matthews_corrcoef(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
    }


def bootstrap_f1(scores, labels, threshold=0.7341, n_resamples=1000, seed=42):
    """
    Bootstrap confidence interval for F1.
    Returns (mean_f1, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    boot_f1s = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        s_boot = scores[idx]
        l_boot = labels[idx]
        p_boot = (s_boot >= threshold).astype(int)
        boot_f1s.append(f1_score(l_boot, p_boot, zero_division=0))
    boot_f1s = np.array(boot_f1s)
    return float(np.mean(boot_f1s)), float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5))


def promote_decision(champion_metrics, challenger_metrics,
                     champ_scores, chall_scores, labels,
                     threshold=0.7341, min_improvement=0.01):
    """
    Decide whether to promote the challenger.

    Promotion criteria (both must be met):
    1. Challenger F1 > Champion F1 + min_improvement (default 1%)
    2. Mann-Whitney U test on per-sample scores: p < 0.05
    3. Bootstrap 95% CI for challenger does NOT overlap champion mean
    """
    champ_f1  = champion_metrics["f1"]
    chall_f1  = challenger_metrics["f1"]
    delta_f1  = chall_f1 - champ_f1

    # Mann-Whitney U on raw scores
    stat, p_value = mannwhitneyu(chall_scores, champ_scores, alternative="greater")

    # Bootstrap CI for challenger
    _, ci_lower, ci_upper = bootstrap_f1(chall_scores, labels)

    promote = (
        delta_f1 >= min_improvement
        and p_value < 0.05
        and ci_lower > champ_f1  # CI lower bound exceeds champion mean
    )

    return {
        "promote":           promote,
        "delta_f1":          float(delta_f1),
        "p_value":           float(p_value),
        "champion_f1":       float(champ_f1),
        "challenger_f1":     float(chall_f1),
        "challenger_ci_low": float(ci_lower),
        "challenger_ci_high":float(ci_upper),
        "reason": (
            "Challenger promoted: meets all criteria." if promote else
            f"Champion retained. Delta F1={delta_f1:.4f} (need >={min_improvement}), "
            f"p={p_value:.4f} (need <0.05), CI lower={ci_lower:.4f} vs champion={champ_f1:.4f}"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Champion/Challenger Evaluation for VulnDetector"
    )
    parser.add_argument("--champion",    required=True,
                        help="Path to current champion checkpoint")
    parser.add_argument("--challenger",  required=True,
                        help="Path to challenger checkpoint from HyperDrive")
    parser.add_argument("--test_path",   required=True,
                        help="Path to test CSV (imbalanced, real-world)")
    parser.add_argument("--cache_path",  required=True,
                        help="Path to DFG cache pkl")
    parser.add_argument("--threshold",   type=float, default=0.7341)
    parser.add_argument("--output",      default="results/champion_challenger.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load both models
    print("Loading champion...")
    champion   = load_model(args.champion,   device)
    print("Loading challenger...")
    challenger = load_model(args.challenger, device)

    # Build test dataloader
    from torch.utils.data import DataLoader
    import pandas as pd
    test_df = pd.read_csv(args.test_path)
    dataset = VulnDataset(test_df, dfg_cache_path=args.cache_path)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    # Get predictions
    print("Running champion inference...")
    champ_scores, champ_preds, labels = get_predictions(
        champion, loader, device, args.threshold
    )
    print("Running challenger inference...")
    chall_scores, chall_preds, _ = get_predictions(
        challenger, loader, device, args.threshold
    )

    # Compute metrics
    champ_metrics = compute_metrics(champ_scores, champ_preds, labels)
    chall_metrics = compute_metrics(chall_scores, chall_preds, labels)

    print("\n=== Champion Metrics ===")
    for k, v in champ_metrics.items():
        print(f"  {k:12}: {v:.4f}")

    print("\n=== Challenger Metrics ===")
    for k, v in chall_metrics.items():
        print(f"  {k:12}: {v:.4f}")

    # Promotion decision
    decision = promote_decision(
        champ_metrics, chall_metrics,
        champ_scores, chall_scores, labels,
        args.threshold,
    )

    print(f"\n=== Promotion Decision ===")
    print(f"  Promote: {decision['promote']}")
    print(f"  Reason : {decision['reason']}")

    # Save full results
    results = {
        "champion":   {"path": args.champion,    "metrics": champ_metrics},
        "challenger": {"path": args.challenger,   "metrics": chall_metrics},
        "decision":   decision,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Exit code 0 = promote, 1 = retain champion
    # Azure DevOps gate reads exit code
    exit(0 if decision["promote"] else 1)


if __name__ == "__main__":
    main()
