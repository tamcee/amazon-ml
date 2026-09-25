"""LightGBM training with grouped split, neg subsampling, full validation."""
import os, pickle, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from .config import cfg, ARTIFACTS_DIR, SEED
from .metrics import f05_macro


FEATURE_COLS = [
    'jaccard_name', 'jaccard_addr', 'levenshtein_name', 'levenshtein_addr',
    'jaro_winkler_name', 'jaro_winkler_addr', 'overlap_name', 'overlap_addr',
    'overlap_ratio_name', 'overlap_ratio_addr', 'len_ratio_name', 'len_ratio_addr',
    'prefix_match_3', 'prefix_match_5', 'exact_name_match', 'exact_addr_match',
    'country_match', 'tfidf_sim', 'emb_name_cos', 'emb_addr_cos',
    'soundex_name_match', 'metaphone_name_match', 'numeric_addr_match',
    'token_sort_ratio_name', 'containment_name', 'containment_addr',
]


def _grouped_split(features_df: pd.DataFrame, val_frac: float = 0.15,
                   min_val: int = 5000) -> tuple:
    """Split by S1 entity groups — no S1 entity appears in both train and val."""
    s1_ids = features_df['s1_id'].unique()
    n_total = len(s1_ids)
    if n_total < 2:
        raise ValueError(f"Need at least 2 unique S1 entities for train/val split, got {n_total}")

    np.random.seed(SEED)
    np.random.shuffle(s1_ids)
    
    # Calculate validation size:
    # 1. Start with target fraction:
    n_val = int(n_total * val_frac)
    # 2. Enforce min_val, but cap it at half the pool (n_total // 2)
    #    so train always retains at least half the entities when n_total < 2 * min_val:
    n_val = max(n_val, min(min_val, n_total // 2))
    # 3. Guard against edge cases: ensure at least 1 entity for val and >=1 entity for train:
    n_val = max(1, min(n_val, n_total - 1))

    val_s1_ids = set(s1_ids[:n_val])
    train_s1_ids = set(s1_ids[n_val:])
    
    if len(train_s1_ids) == 0:
        raise ValueError(
            f"Split error: Train set has 0 S1 entities! Total: {n_total}, n_val: {n_val} "
            f"(val_frac={val_frac}, min_val={min_val})"
        )
    if len(val_s1_ids) == 0:
        raise ValueError(
            f"Split error: Validation set has 0 S1 entities! Total: {n_total}, n_val: {n_val} "
            f"(val_frac={val_frac}, min_val={min_val})"
        )

    train_mask = features_df['s1_id'].isin(train_s1_ids)
    val_mask = features_df['s1_id'].isin(val_s1_ids)
    
    return features_df[train_mask].copy(), features_df[val_mask].copy()


def _subsample_negatives(df: pd.DataFrame, neg_ratio: int = 10) -> pd.DataFrame:
    """Subsample negative pairs to neg_ratio per positive."""
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    
    n_neg_target = len(pos) * neg_ratio
    if len(neg) <= n_neg_target:
        return df
    
    neg_sampled = neg.sample(n=n_neg_target, random_state=SEED)
    result = pd.concat([pos, neg_sampled], ignore_index=True)
    print(f"    Subsampled: {len(pos)} pos + {len(neg_sampled)} neg "
          f"(from {len(neg)} neg, ratio={neg_ratio})")
    return result


def _evaluate_on_full_candidates(model, val_df: pd.DataFrame, 
                                  ground_truth: dict) -> float:
    """Evaluate on FULL unsampled validation candidates.
    
    This is critical: validation F0.5 must be on the complete candidate set,
    not the negative-subsampled training set.
    """
    X_val = val_df[FEATURE_COLS].values
    probas = model.predict(X_val)
    
    # Find best threshold on val
    best_f05 = 0
    best_tau = 0.5
    for tau in np.arange(0.3, 0.99, 0.01):
        predictions = {}
        for _, row in val_df.assign(proba=probas).iterrows():
            s1_id = row['s1_id']
            if row['proba'] >= tau:
                if s1_id not in predictions:
                    predictions[s1_id] = set()
                predictions[s1_id].add(row['s23_id'])
        
        # Ensure all val S1 ids are present
        val_gt = {k: v for k, v in ground_truth.items() if k in set(val_df['s1_id'].unique())}
        for s1_id in val_gt:
            if s1_id not in predictions:
                predictions[s1_id] = set()
        
        score = f05_macro(predictions, val_gt)
        if score > best_f05:
            best_f05 = score
            best_tau = tau
    
    return best_f05, best_tau


def train_model(features_df: pd.DataFrame, ground_truth: dict,
                save_dir: str = None) -> tuple:
    """Train LightGBM model with grouped split and neg subsampling.
    
    Args:
        features_df: DataFrame with s1_id, s23_id, label, and 26 features
        ground_truth: {s1_id: set of true match ids}
        save_dir: directory to save model
    
    Returns:
        (model, best_threshold, val_f05, feature_importance)
    """
    save_dir = save_dir or os.path.join(ARTIFACTS_DIR, 'model')
    os.makedirs(save_dir, exist_ok=True)
    
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    
    print("\n=== Training LightGBM ===")
    print(f"  Total pairs: {len(features_df)} "
          f"(pos: {(features_df['label']==1).sum()}, neg: {(features_df['label']==0).sum()})")
    
    # Grouped split
    train_df, val_df_full = _grouped_split(
        features_df, 
        val_frac=train_cfg.get('val_size', 0.15),
        min_val=train_cfg.get('val_min_entities', 5000)
    )
    print(f"  Train S1 entities: {train_df['s1_id'].nunique()}, "
          f"Val S1 entities: {val_df_full['s1_id'].nunique()}")
    
    # Subsample negatives for training ONLY
    train_sampled = _subsample_negatives(train_df, neg_ratio=train_cfg.get('neg_sample_ratio', 10))
    
    X_train = train_sampled[FEATURE_COLS].values
    y_train = train_sampled['label'].values
    
    # Use FULL val set for early stopping — not subsampled
    X_val = val_df_full[FEATURE_COLS].values
    y_val = val_df_full['label'].values
    
    print(f"  Training: {len(X_train)} pairs, Validation: {len(X_val)} pairs")
    
    # Train
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': model_cfg.get('learning_rate', 0.05),
        'num_leaves': model_cfg.get('num_leaves', 63),
        'max_depth': model_cfg.get('max_depth', -1),
        'min_child_samples': model_cfg.get('min_child_samples', 20),
        'scale_pos_weight': 1.0,  # NOT stacking with neg subsampling
        'seed': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    t0 = time.time()
    callbacks = [
        lgb.early_stopping(model_cfg.get('early_stopping_rounds', 50)),
        lgb.log_evaluation(50),
    ]
    
    model = lgb.train(
        params, train_data,
        num_boost_round=model_cfg.get('n_estimators', 2000),
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks,
    )
    
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f}s, best iteration: {model.best_iteration}")
    
    # Evaluate on FULL validation candidates
    val_f05, best_tau = _evaluate_on_full_candidates(model, val_df_full, ground_truth)
    print(f"  Validation macro F0.5: {val_f05:.4f} (threshold={best_tau:.2f})")
    
    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importance(importance_type='gain')))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print("  Top 10 features by gain:")
    for feat, gain in list(importance.items())[:10]:
        print(f"    {feat}: {gain:.1f}")
    
    # Save
    model_path = os.path.join(save_dir, 'lgbm_model.txt')
    model.save_model(model_path)
    print(f"  Model saved to {model_path}")
    
    meta = {
        'best_threshold': best_tau,
        'val_f05': val_f05,
        'feature_importance': importance,
        'best_iteration': model.best_iteration,
    }
    meta_path = os.path.join(save_dir, 'model_meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    return model, best_tau, val_f05, importance


def load_model(model_dir: str = None) -> tuple:
    """Load a saved model and its metadata."""
    model_dir = model_dir or os.path.join(ARTIFACTS_DIR, 'model')
    model = lgb.Booster(model_file=os.path.join(model_dir, 'lgbm_model.txt'))
    with open(os.path.join(model_dir, 'model_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return model, meta
