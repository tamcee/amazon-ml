"""Create a self-consistent dev cohort for fast iteration.

Usage:
    python -m src.create_dev_cohort --n-entities 1000 --output-dir artifacts/dev_cohort
"""
import argparse, os, random
import pandas as pd
from .config import cfg, TRAIN_DIR, SEED


def create_dev_cohort(n_entities: int = 1000, output_dir: str = None):
    """Sample n S1 entities and extract a self-consistent subset of all sources.
    
    Self-consistent means: only S2/S3 records that are true matches for the sampled
    S1 entities, PLUS a proportional number of distractor S2/S3 records.
    """
    random.seed(SEED)
    output_dir = output_dir or os.path.join(cfg['paths']['artifacts_dir'], 'dev_cohort')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating dev cohort with {n_entities} S1 entities...")
    
    # Load ground truth
    gt = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t',
                     dtype=str, keep_default_na=False)
    
    # Stratified sample: include some singletons and some high-match entities
    gt['n_matches'] = gt['matched_entity_ids'].apply(
        lambda x: len(x.split(',')) if x.strip() else 0
    )
    
    # Sample proportionally to match distribution
    sampled_ids = set()
    for _, group in gt.groupby(gt['n_matches'].clip(upper=5)):
        n_sample = max(1, int(n_entities * len(group) / len(gt)))
        sampled = group.sample(n=min(n_sample, len(group)), random_state=SEED)
        sampled_ids.update(sampled['source1_entity_id'])
    
    # Trim or pad to exact count
    sampled_ids = list(sampled_ids)
    random.shuffle(sampled_ids)
    sampled_ids = set(sampled_ids[:n_entities])
    
    # Get ground truth subset
    gt_sub = gt[gt['source1_entity_id'].isin(sampled_ids)].copy()
    gt_sub.drop(columns=['n_matches'], inplace=True)
    
    # Collect true match IDs
    true_s2_ids = set()
    true_s3_ids = set()
    for _, row in gt_sub.iterrows():
        if row['matched_entity_ids'].strip():
            for mid in row['matched_entity_ids'].split(','):
                mid = mid.strip()
                if mid.startswith('S2-'):
                    true_s2_ids.add(mid)
                elif mid.startswith('S3-'):
                    true_s3_ids.add(mid)
    
    # Load source files
    s1 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), sep='\t',
                     dtype=str, keep_default_na=False)
    s2 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), sep='\t',
                     dtype=str, keep_default_na=False)
    s3 = pd.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), sep='\t',
                     dtype=str, keep_default_na=False)
    
    # S1 subset
    s1_sub = s1[s1['entity_id'].isin(sampled_ids)]
    
    # S2/S3: true matches + proportional distractors
    s2_true = s2[s2['entity_id'].isin(true_s2_ids)]
    s3_true = s3[s3['entity_id'].isin(true_s3_ids)]
    
    # Add ~5x distractors
    s2_rest = s2[~s2['entity_id'].isin(true_s2_ids)]
    s3_rest = s3[~s3['entity_id'].isin(true_s3_ids)]
    n_distract_s2 = min(len(true_s2_ids) * 5, len(s2_rest))
    n_distract_s3 = min(len(true_s3_ids) * 5, len(s3_rest))
    
    s2_distract = s2_rest.sample(n=n_distract_s2, random_state=SEED) if n_distract_s2 > 0 else s2_rest.head(0)
    s3_distract = s3_rest.sample(n=n_distract_s3, random_state=SEED) if n_distract_s3 > 0 else s3_rest.head(0)
    
    s2_sub = pd.concat([s2_true, s2_distract], ignore_index=True)
    s3_sub = pd.concat([s3_true, s3_distract], ignore_index=True)
    
    # Save
    s1_sub.to_csv(os.path.join(output_dir, 'train_source1.tsv'), sep='\t', index=False)
    s2_sub.to_csv(os.path.join(output_dir, 'train_source2.tsv'), sep='\t', index=False)
    s3_sub.to_csv(os.path.join(output_dir, 'train_source3.tsv'), sep='\t', index=False)
    gt_sub.to_csv(os.path.join(output_dir, 'train_ground_truth.tsv'), sep='\t', index=False)
    
    print(f"Dev cohort saved to {output_dir}:")
    print(f"  S1: {len(s1_sub)} entities")
    print(f"  S2: {len(s2_sub)} records ({len(s2_true)} true matches + {len(s2_distract)} distractors)")
    print(f"  S3: {len(s3_sub)} records ({len(s3_true)} true matches + {len(s3_distract)} distractors)")
    print(f"  GT: {len(gt_sub)} entries")
    return output_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-entities', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()
    create_dev_cohort(args.n_entities, args.output_dir)
