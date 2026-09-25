"""Blocking — multi-strategy inverted index with max-block-size guard."""
import re
from collections import defaultdict, Counter
import jellyfish
import pandas as pd


def _soundex_safe(token: str) -> str:
    """Compute soundex, returning empty string on failure."""
    try:
        if len(token) >= 2 and token.isalpha():
            return jellyfish.soundex(token)
    except Exception:
        pass
    return ''


def _metaphone_safe(token: str) -> str:
    """Compute metaphone, returning empty string on failure."""
    try:
        if len(token) >= 2 and token.isalpha():
            return jellyfish.metaphone(token)
    except Exception:
        pass
    return ''


def generate_blocking_keys(norm_name: str, norm_addr: str) -> set:
    """Generate blocking keys for one entity from normalized name and address."""
    keys = set()
    name_tokens = norm_name.split() if norm_name else []
    addr_tokens = norm_addr.split() if norm_addr else []
    
    # Strategy 1: Sorted 3-token name key
    if len(name_tokens) >= 3:
        sorted_3 = '_'.join(sorted(name_tokens[:3]))
        keys.add(f'n3:{sorted_3}')
    elif len(name_tokens) == 2:
        sorted_2 = '_'.join(sorted(name_tokens[:2]))
        keys.add(f'n2:{sorted_2}')
    
    # Strategy 2: Phonetic pair keys (first 2 tokens)
    phonetic_codes = []
    for t in name_tokens[:3]:
        sx = _soundex_safe(t)
        if sx:
            phonetic_codes.append(sx)
    if len(phonetic_codes) >= 2:
        keys.add(f'ph:{phonetic_codes[0]}_{phonetic_codes[1]}')
    
    # Strategy 3: Name prefix (first 5 chars)
    if len(norm_name) >= 5:
        keys.add(f'pfx:{norm_name[:5]}')
    elif len(norm_name) >= 3:
        keys.add(f'pfx:{norm_name[:3]}')
    
    # Strategy 4: Address distinctive tokens
    for t in addr_tokens:
        if len(t) >= 4 and not t.isdigit():
            keys.add(f'addr:{t}')
    
    # Strategy 5: Single-word phonetic keys
    for t in name_tokens:
        sx = _soundex_safe(t)
        if sx:
            keys.add(f'sph:{sx}')
        mp = _metaphone_safe(t)
        if mp:
            keys.add(f'mph:{mp}')
    
    # Strategy 6: Distinctive single tokens ≥4 chars from name
    for t in name_tokens:
        if len(t) >= 4:
            keys.add(f'tok:{t}')
    
    # Strategy 7: All sorted bigram pairs from name tokens
    if len(name_tokens) >= 2:
        for i in range(len(name_tokens)):
            for j in range(i + 1, min(i + 4, len(name_tokens))):
                pair = '_'.join(sorted([name_tokens[i], name_tokens[j]]))
                keys.add(f'bg:{pair}')
    
    # Strategy 8: Address numeric × substantive-token
    addr_nums = [t for t in addr_tokens if t.isdigit()]
    addr_subst = [t for t in addr_tokens if len(t) >= 4 and not t.isdigit()]
    for num in addr_nums[:2]:
        for subst in addr_subst[:2]:
            keys.add(f'an:{num}_{subst}')
    
    return keys


def build_inverted_index(df: pd.DataFrame, max_block_size: int = 2500) -> dict:
    """Build inverted index from S2/S3 records.
    
    Args:
        df: DataFrame with entity_id, norm_business_name, norm_business_address
        max_block_size: drop keys whose posting list exceeds this size
    
    Returns:
        dict mapping blocking key -> set of entity_ids
    """
    index = defaultdict(set)
    
    for _, row in df.iterrows():
        eid = row['entity_id']
        keys = generate_blocking_keys(
            row.get('norm_business_name', ''),
            row.get('norm_business_address', '')
        )
        for k in keys:
            index[k].add(eid)
    
    # Apply max-block-size guard
    oversized = {k for k, v in index.items() if len(v) > max_block_size}
    if oversized:
        print(f"  Dropping {len(oversized)} oversized blocking keys (>{max_block_size} entries)")
        for k in oversized:
            del index[k]
    
    print(f"  Inverted index: {len(index)} keys, "
          f"{sum(len(v) for v in index.values())} total postings")
    return dict(index)


def find_candidates(s1_df: pd.DataFrame, index: dict,
                    top_k: int = 100) -> dict:
    """Find candidate matches for each S1 entity using the inverted index.
    
    Args:
        s1_df: S1 DataFrame with norm_business_name, norm_business_address
        index: inverted index from build_inverted_index
        top_k: max candidates per S1 entity (by overlap count)
    
    Returns:
        dict: {s1_entity_id: set of candidate entity_ids}
    """
    candidates = {}
    
    for _, row in s1_df.iterrows():
        s1_id = row['entity_id']
        keys = generate_blocking_keys(
            row.get('norm_business_name', ''),
            row.get('norm_business_address', '')
        )
        
        # Count overlaps
        overlap_counts = Counter()
        for k in keys:
            if k in index:
                for cand_id in index[k]:
                    overlap_counts[cand_id] += 1
        
        # Top-k by overlap count
        if overlap_counts:
            top_cands = {eid for eid, _ in overlap_counts.most_common(top_k)}
        else:
            top_cands = set()
        
        candidates[s1_id] = top_cands
    
    return candidates


def run_blocking(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                 max_block_size: int = 2500, top_k: int = 100,
                 ground_truth: dict = None) -> dict:
    """Full blocking pipeline: build index from S2+S3, find candidates for S1.
    
    Args:
        s1_df, s2_df, s3_df: Normalized DataFrames
        max_block_size: max posting list size
        top_k: max candidates per S1 entity
        ground_truth: optional {s1_id: set(match_ids)} for recall evaluation
    
    Returns:
        dict: {s1_entity_id: set of candidate entity_ids}
    """
    print("Building inverted index...")
    s23 = pd.concat([s2_df, s3_df], ignore_index=True)
    index = build_inverted_index(s23, max_block_size=max_block_size)
    
    print(f"Finding candidates for {len(s1_df)} S1 entities...")
    candidates = find_candidates(s1_df, index, top_k=top_k)
    
    # Stats
    n_with_cands = sum(1 for v in candidates.values() if v)
    avg_cands = sum(len(v) for v in candidates.values()) / max(len(candidates), 1)
    print(f"  {n_with_cands}/{len(candidates)} S1 entities have candidates")
    print(f"  Avg candidates per entity: {avg_cands:.1f}")
    
    # Evaluate recall if ground truth provided
    if ground_truth:
        from .metrics import blocking_recall
        recall_info = blocking_recall(candidates, ground_truth)
        print(f"  Blocking recall: {recall_info['recall']:.4f} "
              f"({recall_info['found_pairs']}/{recall_info['total_true_pairs']} true pairs found)")
        if recall_info['missed_examples']:
            print(f"  Sample missed pairs (first 5):")
            for s1, mid in recall_info['missed_examples'][:5]:
                print(f"    {s1} -> {mid}")
    
    return candidates
