"""Test inference — end-to-end from raw test data to submission files."""
import os, time
import pandas as pd
import numpy as np
from .config import cfg, TEST_DIR, OUTPUT_DIR, ARTIFACTS_DIR, DEVICE
from .normalize import normalize_all_sources
from .blocking import run_blocking
from .embeddings import EmbeddingManager
from .features import FeatureExtractor
from .train import load_model, FEATURE_COLS
from .validate import validate_submission


def run_inference(model=None, threshold: float = None, cap: int = None,
                  test_dir: str = None, output_dir: str = None):
    """Run full inference pipeline on test data.
    
    Args:
        model: trained LightGBM model (or None to load from disk)
        threshold: classification threshold (or None to load from model meta)
        cap: max matches per entity (None = no cap)
        test_dir: test data directory
        output_dir: output directory
    """
    test_dir = test_dir or TEST_DIR
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    t_start = time.time()
    
    # Load model if needed
    if model is None:
        model, meta = load_model()
        if threshold is None:
            threshold = meta['best_threshold']
        print(f"Loaded model (best_iter={meta['best_iteration']}, threshold={threshold:.2f})")
    elif threshold is None:
        threshold = 0.5
    
    # 1. Load test data
    print("\n=== Loading test data ===")
    s1 = pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t',
                     dtype=str, keep_default_na=False)
    s2 = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t',
                     dtype=str, keep_default_na=False)
    s3 = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t',
                     dtype=str, keep_default_na=False)
    print(f"  S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}")
    
    # 2. Normalize
    print("\n=== Normalizing ===")
    cache_dir = os.path.join(ARTIFACTS_DIR, 'test_normalized')
    s1, s2, s3 = normalize_all_sources(s1, s2, s3, cache_dir=cache_dir)
    
    # 3. Blocking
    print("\n=== Blocking ===")
    blocking_cfg = cfg['blocking']
    candidates = run_blocking(
        s1, s2, s3,
        max_block_size=blocking_cfg['max_block_size'],
        top_k=blocking_cfg['top_k_per_entity'],
    )
    
    # Save candidate_pairs.tsv
    _write_candidate_pairs(candidates, output_dir)
    
    # 4. Embeddings
    print("\n=== Embeddings ===")
    emb_mgr = EmbeddingManager(
        cache_dir=os.path.join(ARTIFACTS_DIR, 'test_embeddings'),
        batch_size=cfg['features']['embedding_batch_size'],
        device=DEVICE,
    )
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    import gc
    gc.collect()
    name_embs, addr_embs = emb_mgr.embed_candidate_pairs(s1, s23, candidates)
    
    # Flush GPU cache after test embedding generation
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    
    # 5. Features
    print("\n=== Feature extraction ===")
    feat_ext = FeatureExtractor(max_features=cfg['features']['tfidf_max_features'])
    
    # Fit TF-IDF on test data series without full DataFrame copy
    name_col = 'norm_business_name' if 'norm_business_name' in s1.columns else 'business_name'
    all_names = pd.concat([s1[name_col], s23[name_col]], ignore_index=True)
    all_ids = pd.concat([s1['entity_id'], s23['entity_id']], ignore_index=True)
    feat_ext.fit_tfidf(all_names, all_ids)
    del all_names, all_ids
    gc.collect()
    
    features_df = feat_ext.extract_all_features(
        s1, s23, candidates, name_embs, addr_embs
    )
    # Free raw text embeddings, s23, and candidates as features are computed
    del name_embs, addr_embs, s23, candidates
    gc.collect()
    
    # 6. Score
    print("\n=== Scoring ===")
    X = features_df[FEATURE_COLS].values
    probas = model.predict(X)
    features_df['proba'] = probas
    
    # 7. Apply threshold
    print(f"  Applying threshold={threshold:.2f}, cap={cap}")
    predictions = {}
    for s1_id in s1['entity_id'].unique():
        predictions[s1_id] = set()
    
    above = features_df[features_df['proba'] >= threshold].copy()
    if cap is not None:
        above = above.sort_values('proba', ascending=False)
        above = above.groupby('s1_id').head(cap)
    
    for row in above[['s1_id', 's23_id']].itertuples(index=False):
        predictions[row.s1_id].add(row.s23_id)
    
    # Stats
    n_matched = sum(1 for v in predictions.values() if v)
    n_singleton = sum(1 for v in predictions.values() if not v)
    avg_matches = sum(len(v) for v in predictions.values()) / max(len(predictions), 1)
    print(f"  Matched: {n_matched}, Singletons: {n_singleton}, Avg matches: {avg_matches:.2f}")
    
    # 8. Write output
    _write_matching_results(predictions, output_dir)
    
    elapsed = time.time() - t_start
    print(f"\n=== Inference complete in {elapsed:.1f}s ===")
    
    # 9. Validate
    print("\n=== Validation ===")
    exit_code, _ = validate_submission(
        os.path.join(output_dir, 'matching_results.tsv'),
        os.path.join(output_dir, 'candidate_pairs.tsv'),
        test_dir,
    )
    if exit_code == 0:
        print("PASS — submission is valid!")
    else:
        print("FAIL — see errors above")
    
    return predictions


def _write_matching_results(predictions: dict, output_dir: str):
    """Write matching_results.tsv."""
    path = os.path.join(output_dir, 'matching_results.tsv')
    rows = []
    for s1_id in sorted(predictions.keys()):
        matched = ','.join(sorted(predictions[s1_id]))
        rows.append(f"{s1_id}\t{matched}")
    
    with open(path, 'w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        f.write('\n'.join(rows) + '\n')
    
    print(f"  Wrote {len(rows)} rows to {path}")


def _write_candidate_pairs(candidates: dict, output_dir: str):
    """Write candidate_pairs.tsv."""
    path = os.path.join(output_dir, 'candidate_pairs.tsv')
    rows = []
    for s1_id in sorted(candidates.keys()):
        cands = ','.join(sorted(candidates[s1_id]))
        rows.append(f"{s1_id}\t{cands}")
    
    with open(path, 'w') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        f.write('\n'.join(rows) + '\n')
    
    print(f"  Wrote {len(rows)} candidate rows to {path}")
