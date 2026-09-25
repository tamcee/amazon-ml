"""Evaluation metrics — F0.5 macro scorer and blocking recall."""
import numpy as np


def f05_single(predicted: set, actual: set) -> float:
    """F0.5 for a single entity. Singletons: 1.0 if both empty, 0.0 otherwise."""
    if not actual and not predicted:
        return 1.0
    if not actual or not predicted:
        return 0.0
    tp = len(predicted & actual)
    precision = tp / len(predicted)
    recall = tp / len(actual)
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def f05_macro(predictions: dict, ground_truth: dict) -> float:
    """Macro-averaged F0.5 across all entities.
    
    Args:
        predictions: {s1_id: set of matched ids}
        ground_truth: {s1_id: set of matched ids}
    Returns:
        Macro F0.5 score
    """
    scores = []
    for s1_id, actual in ground_truth.items():
        predicted = predictions.get(s1_id, set())
        scores.append(f05_single(predicted, actual))
    return float(np.mean(scores)) if scores else 0.0


def blocking_recall(candidates: dict, ground_truth: dict) -> dict:
    """Compute blocking recall — fraction of true matches present in candidates.
    
    Args:
        candidates: {s1_id: set of candidate ids}
        ground_truth: {s1_id: set of true match ids}
    Returns:
        dict with 'recall', 'total_true_pairs', 'found_pairs', 'missed_pairs'
    """
    total = 0
    found = 0
    missed_examples = []
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        cands = candidates.get(s1_id, set())
        for mid in true_matches:
            total += 1
            if mid in cands:
                found += 1
            else:
                if len(missed_examples) < 20:
                    missed_examples.append((s1_id, mid))
    recall = found / total if total > 0 else 1.0
    return {
        'recall': recall,
        'total_true_pairs': total,
        'found_pairs': found,
        'missed_pairs': total - found,
        'missed_examples': missed_examples,
    }


def singleton_accuracy(predictions: dict, ground_truth: dict) -> float:
    """Fraction of true singletons correctly predicted as empty."""
    total = 0
    correct = 0
    for s1_id, actual in ground_truth.items():
        if not actual:  # true singleton
            total += 1
            if not predictions.get(s1_id, set()):
                correct += 1
    return correct / total if total > 0 else 1.0
