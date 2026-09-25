"""Threshold and cap optimization for macro F0.5."""
import numpy as np
import pandas as pd
from .metrics import f05_macro, singleton_accuracy


def sweep_threshold(val_df: pd.DataFrame, ground_truth: dict,
                    model=None, probas: np.ndarray = None,
                    min_tau: float = 0.3, max_tau: float = 0.99,
                    step: float = 0.01, caps: list = None) -> dict:
    """Sweep threshold and optional cap to find optimal macro F0.5.
    
    Args:
        val_df: Full validation candidate DataFrame with s1_id, s23_id
        ground_truth: {s1_id: set of true match ids}
        model: LightGBM model (or None if probas provided)
        probas: pre-computed probabilities (or None to compute from model)
        min_tau, max_tau, step: threshold search range
        caps: list of max-match caps to try (None = no cap)
    
    Returns:
        dict with optimal settings and scores
    """
    from .train import FEATURE_COLS
    
    if probas is None:
        X = val_df[FEATURE_COLS].values
        probas = model.predict(X)
    
    val_df = val_df.copy()
    val_df['proba'] = probas
    
    # Get val S1 IDs
    val_s1_ids = set(val_df['s1_id'].unique())
    val_gt = {k: v for k, v in ground_truth.items() if k in val_s1_ids}
    
    if caps is None:
        caps = [None]  # No cap
    
    best_score = 0
    best_config = {}
    results = []
    
    for tau in np.arange(min_tau, max_tau + step/2, step):
        tau = round(tau, 4)
        for cap in caps:
            # Build predictions
            predictions = {}
            for s1_id in val_s1_ids:
                predictions[s1_id] = set()
            
            # Group by s1_id, filter by threshold, apply cap
            above = val_df[val_df['proba'] >= tau].copy()
            if cap is not None:
                above = above.sort_values('proba', ascending=False)
                above = above.groupby('s1_id').head(cap)
            
            for _, row in above.iterrows():
                predictions[row['s1_id']].add(row['s23_id'])
            
            score = f05_macro(predictions, val_gt)
            sing_acc = singleton_accuracy(predictions, val_gt)
            
            results.append({
                'tau': tau, 'cap': cap, 'f05': score, 'singleton_acc': sing_acc
            })
            
            if score > best_score:
                best_score = score
                best_config = {'tau': tau, 'cap': cap, 'f05': score, 'singleton_acc': sing_acc}
    
    print(f"\n  Threshold sweep results:")
    print(f"  Best: tau={best_config['tau']:.2f}, cap={best_config['cap']}, "
          f"F0.5={best_config['f05']:.4f}, singleton_acc={best_config['singleton_acc']:.4f}")
    
    return {
        'best': best_config,
        'all_results': pd.DataFrame(results),
    }
