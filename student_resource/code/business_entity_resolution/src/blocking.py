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
    # Limited to first 10 tokens to avoid unbounded key explosion on noisy/concatenated names
    for t in name_tokens[:10]:
        if len(t) >= 4:
            keys.add(f'tok:{t}')
    
    # Strategy 7: All sorted bigram pairs from name tokens
    # Bounded window: pairwise combinations within a sliding window of 4 tokens, limited to first 10 tokens
    bounded_name_tokens = name_tokens[:10]
    if len(bounded_name_tokens) >= 2:
        for i in range(len(bounded_name_tokens)):
            for j in range(i + 1, min(i + 4, len(bounded_name_tokens))):
                pair = '_'.join(sorted([bounded_name_tokens[i], bounded_name_tokens[j]]))
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
    
    Memory optimizations:
    1. Encodes entity_ids as compact integer indices (small ints) instead of storing
       repeated full string references across posting lists.
    2. Enforces max_block_size incrementally: as soon as a key's block exceeds
       max_block_size, the key is pruned and added to a blocked_keys blacklist,
       preventing peak memory spikes from oversized keys.
    3. Fast iteration using itertuples() instead of iterrows().
    
    Args:
        df: DataFrame with entity_id, norm_business_name, norm_business_address
        max_block_size: drop keys whose posting list exceeds this size
    
    Returns:
        dict mapping blocking key -> set of entity_ids (or integer IDs with mapping attached)
    """
    import gc
    
    unique_eids = df['entity_id'].unique()
    eid_to_int = {eid: idx for idx, eid in enumerate(unique_eids)}
    int_to_eid = list(unique_eids)
    
    index = defaultdict(set)
    blocked_keys = set()
    
    # Extract only required columns for faster, lighter itertuples iteration
    cols = ['entity_id', 'norm_business_name', 'norm_business_address']
    for row in df[cols].itertuples(index=False):
        eid_int = eid_to_int[row[0]]
        keys = generate_blocking_keys(row[1] or '', row[2] or '')
        for k in keys:
            if k in blocked_keys:
                continue
            posting_set = index[k]
            posting_set.add(eid_int)
            if len(posting_set) > max_block_size:
                del index[k]
                blocked_keys.add(k)
    
    dropped_count = len(blocked_keys)
    if dropped_count > 0:
        print(f"  Incrementally pruned {dropped_count} oversized blocking keys (>{max_block_size} entries)")
    
    total_postings = sum(len(v) for v in index.values())
    print(f"  Inverted index: {len(index)} keys, {total_postings} total postings")
    
    # Attach int_to_eid lookup to the index dict metadata for find_candidates
    result = dict(index)
    result['_int_to_eid'] = int_to_eid
    
    del eid_to_int, blocked_keys
    gc.collect()
    return result


def find_candidates(s1_df: pd.DataFrame, index: dict,
                    top_k: int = 100) -> dict:
    """Find candidate matches for each S1 entity using the inverted index.
    
    Optimized to iterate via itertuples() and decode compact integer postings
    back to original string entity_ids.
    
    Args:
        s1_df: S1 DataFrame with norm_business_name, norm_business_address
        index: inverted index from build_inverted_index
        top_k: max candidates per S1 entity (by overlap count)
    
    Returns:
        dict: {s1_entity_id: set of candidate entity_ids}
    """
    int_to_eid = index.get('_int_to_eid', None)
    candidates = {}
    cols = ['entity_id', 'norm_business_name', 'norm_business_address']
    
    for row in s1_df[cols].itertuples(index=False):
        s1_id = row[0]
        keys = generate_blocking_keys(row[1] or '', row[2] or '')
        
        # Count overlaps
        overlap_counts = Counter()
        for k in keys:
            if k in index:
                for cand_id in index[k]:
                    overlap_counts[cand_id] += 1
        
        # Top-k by overlap count
        if overlap_counts:
            if int_to_eid is not None:
                top_cands = {int_to_eid[cand_id] for cand_id, _ in overlap_counts.most_common(top_k)}
            else:
                top_cands = {cand_id for cand_id, _ in overlap_counts.most_common(top_k)}
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
    import gc
    
    print("Building inverted index...")
    # Concatenate S2 and S3 for indexing
    s23 = pd.concat([s2_df, s3_df], ignore_index=True)
    index = build_inverted_index(s23, max_block_size=max_block_size)
    
    # Free concatenated DataFrame immediately after index is built
    del s23
    gc.collect()
    
    print(f"Finding candidates for {len(s1_df)} S1 entities...")
    candidates = find_candidates(s1_df, index, top_k=top_k)
    
    # Free index immediately after candidates are generated
    del index
    gc.collect()
    
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
