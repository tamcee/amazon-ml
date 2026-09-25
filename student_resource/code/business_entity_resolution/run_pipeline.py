#!/usr/bin/env python3
"""Business Entity Resolution Pipeline — end-to-end driver.

Usage:
    python run_pipeline.py --stage all          # full pipeline
    python run_pipeline.py --stage train        # train only
    python run_pipeline.py --stage infer        # inference only
    python run_pipeline.py --stage dev_cohort   # create dev cohort
"""
import argparse, os, sys, time
import pandas as pd


def load_ground_truth(gt_path: str) -> dict:
    """Load ground truth into {s1_id: set of match ids} dict."""
    gt = pd.read_csv(gt_path, sep='\t', dtype=str, keep_default_na=False)
    result = {}
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        matched = row['matched_entity_ids'].strip()
        if matched:
            result[s1_id] = set(matched.split(','))
        else:
            result[s1_id] = set()
    return result


def stage_dev_cohort(args):
    """Create a dev cohort for fast iteration."""
    from src.create_dev_cohort import create_dev_cohort
    create_dev_cohort(n_entities=args.dev_size)


def stage_train(args):
    """Full training pipeline: normalize → block → embed → features → train → threshold."""
    from src.config import cfg, TRAIN_DIR, ARTIFACTS_DIR, DEVICE
    from src.normalize import normalize_all_sources
    from src.blocking import run_blocking
    from src.embeddings import EmbeddingManager
    from src.features import FeatureExtractor
    from src.train import train_model
    from src.threshold import sweep_threshold
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE")
    print("="*60)
    
    train_dir = os.environ.get('TRAIN_DIR', TRAIN_DIR)
    print(f"\nData directory: {train_dir}")
    
    # Load data
    print("\n=== Loading training data ===")
    s1 = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t',
                     dtype=str, keep_default_na=False)
    s2 = pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t',
                     dtype=str, keep_default_na=False)
    s3 = pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t',
                     dtype=str, keep_default_na=False)
    gt = load_ground_truth(os.path.join(train_dir, 'train_ground_truth.tsv'))
    print(f"  S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}, GT: {len(gt)}")
    
    # Normalize
    print("\n=== Normalization ===")
    cache_dir = os.path.join(ARTIFACTS_DIR, 'train_normalized')
    s1, s2, s3 = normalize_all_sources(s1, s2, s3, cache_dir=cache_dir)
    
    # Blocking
    print("\n=== Blocking ===")
    blocking_cfg = cfg['blocking']
    candidates = run_blocking(
        s1, s2, s3,
        max_block_size=blocking_cfg['max_block_size'],
        top_k=blocking_cfg['top_k_per_entity'],
        ground_truth=gt,
    )
    
    # Embeddings
    print("\n=== Embeddings ===")
    emb_mgr = EmbeddingManager(
        cache_dir=os.path.join(ARTIFACTS_DIR, 'train_embeddings'),
        batch_size=cfg['features']['embedding_batch_size'],
        device=DEVICE,
    )
    s23 = pd.concat([s2, s3], ignore_index=True)
    name_embs, addr_embs = emb_mgr.embed_candidate_pairs(s1, s23, candidates)
    
    # Features
    print("\n=== Feature Extraction ===")
    feat_ext = FeatureExtractor(max_features=cfg['features']['tfidf_max_features'])
    
    # Fit TF-IDF
    all_df = pd.concat([s1, s23], ignore_index=True)
    name_col = 'norm_business_name' if 'norm_business_name' in all_df.columns else 'business_name'
    feat_ext.fit_tfidf(all_df[name_col], all_df['entity_id'])
    
    features_df = feat_ext.extract_all_features(
        s1, s23, candidates, name_embs, addr_embs
    )
    
    # Add labels
    labels = []
    for _, row in features_df.iterrows():
        s1_id = row['s1_id']
        s23_id = row['s23_id']
        true_matches = gt.get(s1_id, set())
        labels.append(1 if s23_id in true_matches else 0)
    features_df['label'] = labels
    
    print(f"  Labels: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    # Train
    model, best_tau, val_f05, importance = train_model(features_df, gt)
    
    # Threshold sweep (on training val split)
    print("\n=== Threshold Optimization ===")
    # Re-split to get val set for sweep
    from src.train import _grouped_split
    _, val_df = _grouped_split(features_df)
    sweep_results = sweep_threshold(val_df, gt, model=model)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Best threshold: {sweep_results['best']['tau']:.2f}")
    print(f"  Validation F0.5: {sweep_results['best']['f05']:.4f}")
    print(f"{'='*60}")


def stage_infer(args):
    """Run inference on test data."""
    from src.infer import run_inference
    
    print("\n" + "="*60)
    print("INFERENCE PIPELINE")
    print("="*60)
    
    run_inference()


def main():
    parser = argparse.ArgumentParser(description='Business Entity Resolution Pipeline')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'train', 'infer', 'dev_cohort'],
                        help='Pipeline stage to run')
    parser.add_argument('--dev-size', type=int, default=1000,
                        help='Number of S1 entities for dev cohort')
    args = parser.parse_args()
    
    t_start = time.time()
    
    if args.stage == 'dev_cohort':
        stage_dev_cohort(args)
    elif args.stage == 'train':
        stage_train(args)
    elif args.stage == 'infer':
        stage_infer(args)
    elif args.stage == 'all':
        stage_train(args)
        stage_infer(args)
    
    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
