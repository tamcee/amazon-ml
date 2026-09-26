"""Pairwise feature extraction — 26 features with bounded sparse TF-IDF."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def _jaccard(s1_tokens: set, s2_tokens: set) -> float:
    if not s1_tokens and not s2_tokens:
        return 1.0
    if not s1_tokens or not s2_tokens:
        return 0.0
    return len(s1_tokens & s2_tokens) / len(s1_tokens | s2_tokens)


def _overlap_count(s1_tokens: set, s2_tokens: set) -> int:
    return len(s1_tokens & s2_tokens)


def _overlap_ratio(s1_tokens: set, s2_tokens: set) -> float:
    if not s1_tokens or not s2_tokens:
        return 0.0
    return len(s1_tokens & s2_tokens) / min(len(s1_tokens), len(s2_tokens))


def _len_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def _prefix_match(a: str, b: str, n: int) -> float:
    if len(a) < n or len(b) < n:
        return 0.0
    return 1.0 if a[:n] == b[:n] else 0.0


def _containment(a_tokens: set, b_tokens: set) -> float:
    """Fraction of a's tokens contained in b."""
    if not a_tokens:
        return 1.0 if not b_tokens else 0.0
    return len(a_tokens & b_tokens) / len(a_tokens)


def _numeric_tokens(text: str) -> set:
    return {t for t in text.split() if t.isdigit()}


def _cosine_f16(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two float16 vectors."""
    if a is None or b is None:
        return 0.0
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class FeatureExtractor:
    """Extracts 26 pairwise features for candidate entity pairs."""
    
    def __init__(self, max_features: int = 100000):
        self.max_features = max_features
        self.tfidf = None
        self._tfidf_matrix = None
        self._tfidf_id_to_idx = {}
    
    def fit_tfidf(self, all_names: pd.Series, all_ids: pd.Series):
        """Fit TF-IDF vectorizer on all normalized names."""
        print(f"  Fitting TF-IDF on {len(all_names)} texts (max_features={self.max_features})...")
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            analyzer='char_wb',
            ngram_range=(3, 4),
            sublinear_tf=True,
            dtype=np.float32,
        )
        texts = all_names.fillna('').tolist()
        self._tfidf_matrix = self.tfidf.fit_transform(texts)
        self._tfidf_id_to_idx = {eid: idx for idx, eid in enumerate(all_ids.tolist())}
        print(f"  TF-IDF matrix shape: {self._tfidf_matrix.shape}")
    
    def tfidf_similarity(self, id_a: str, id_b: str) -> float:
        """Compute TF-IDF cosine similarity between two entities by ID."""
        idx_a = self._tfidf_id_to_idx.get(id_a)
        idx_b = self._tfidf_id_to_idx.get(id_b)
        if idx_a is None or idx_b is None:
            return 0.0
        vec_a = self._tfidf_matrix[idx_a]
        vec_b = self._tfidf_matrix[idx_b]
        sim = (vec_a @ vec_b.T).toarray()[0, 0]
        return float(sim)
    
    def extract_pair_features(self, s1_row: dict, s23_row: dict,
                               name_embs: dict = None, addr_embs: dict = None) -> dict:
        """Extract 26 features for a single (s1, s23) pair."""
        s1_name = s1_row.get('norm_business_name', '')
        s23_name = s23_row.get('norm_business_name', '')
        s1_addr = s1_row.get('norm_business_address', '')
        s23_addr = s23_row.get('norm_business_address', '')
        s1_country = str(s1_row.get('country', '')).lower().strip()
        s23_country = str(s23_row.get('country', '')).lower().strip()
        
        s1_name_tok = set(s1_name.split()) if s1_name else set()
        s23_name_tok = set(s23_name.split()) if s23_name else set()
        s1_addr_tok = set(s1_addr.split()) if s1_addr else set()
        s23_addr_tok = set(s23_addr.split()) if s23_addr else set()
        
        s1_id = s1_row.get('entity_id', '')
        s23_id = s23_row.get('entity_id', '')
        
        features = {
            'jaccard_name': _jaccard(s1_name_tok, s23_name_tok),
            'jaccard_addr': _jaccard(s1_addr_tok, s23_addr_tok),
            'levenshtein_name': fuzz.ratio(s1_name, s23_name) / 100.0,
            'levenshtein_addr': fuzz.ratio(s1_addr, s23_addr) / 100.0,
            'jaro_winkler_name': jellyfish.jaro_winkler_similarity(s1_name, s23_name) if s1_name and s23_name else 0.0,
            'jaro_winkler_addr': jellyfish.jaro_winkler_similarity(s1_addr, s23_addr) if s1_addr and s23_addr else 0.0,
            'overlap_name': _overlap_count(s1_name_tok, s23_name_tok),
            'overlap_addr': _overlap_count(s1_addr_tok, s23_addr_tok),
            'overlap_ratio_name': _overlap_ratio(s1_name_tok, s23_name_tok),
            'overlap_ratio_addr': _overlap_ratio(s1_addr_tok, s23_addr_tok),
            'len_ratio_name': _len_ratio(s1_name, s23_name),
            'len_ratio_addr': _len_ratio(s1_addr, s23_addr),
            'prefix_match_3': _prefix_match(s1_name, s23_name, 3),
            'prefix_match_5': _prefix_match(s1_name, s23_name, 5),
            'exact_name_match': 1.0 if s1_name == s23_name and s1_name else 0.0,
            'exact_addr_match': 1.0 if s1_addr == s23_addr and s1_addr else 0.0,
            'country_match': 1.0 if s1_country == s23_country and s1_country else 0.0,
            'tfidf_sim': self.tfidf_similarity(s1_id, s23_id) if self.tfidf else 0.0,
            'soundex_name_match': 0.0,
            'metaphone_name_match': 0.0,
            'numeric_addr_match': 0.0,
            'token_sort_ratio_name': fuzz.token_sort_ratio(s1_name, s23_name) / 100.0 if s1_name and s23_name else 0.0,
            'containment_name': _containment(s1_name_tok, s23_name_tok),
            'containment_addr': _containment(s1_addr_tok, s23_addr_tok),
        }
        
        # Soundex/Metaphone on first token
        if s1_name_tok and s23_name_tok:
            s1_first = sorted(s1_name_tok)[0]
            s23_first = sorted(s23_name_tok)[0]
            try:
                if s1_first.isalpha() and s23_first.isalpha():
                    features['soundex_name_match'] = 1.0 if jellyfish.soundex(s1_first) == jellyfish.soundex(s23_first) else 0.0
                    features['metaphone_name_match'] = 1.0 if jellyfish.metaphone(s1_first) == jellyfish.metaphone(s23_first) else 0.0
            except Exception:
                pass
        
        # Numeric address match
        s1_nums = _numeric_tokens(s1_addr)
        s23_nums = _numeric_tokens(s23_addr)
        if s1_nums and s23_nums:
            features['numeric_addr_match'] = len(s1_nums & s23_nums) / max(len(s1_nums), len(s23_nums))
        
        # Embedding cosine similarities
        if name_embs:
            features['emb_name_cos'] = _cosine_f16(
                name_embs.get(s1_id), name_embs.get(s23_id)
            )
        else:
            features['emb_name_cos'] = 0.0
        
        if addr_embs:
            features['emb_addr_cos'] = _cosine_f16(
                addr_embs.get(s1_id), addr_embs.get(s23_id)
            )
        else:
            features['emb_addr_cos'] = 0.0
        
        return features
    
    def extract_all_features(self, s1_df: pd.DataFrame, s23_df: pd.DataFrame,
                              candidates: dict,
                              name_embs: dict = None, addr_embs: dict = None,
                              progress_interval: int = 10000) -> pd.DataFrame:
        """Extract features for all candidate pairs.
        
        Args:
            s1_df: S1 DataFrame (normalized)
            s23_df: S2+S3 DataFrame (normalized)
            candidates: {s1_id: set of candidate s23 ids}
            name_embs, addr_embs: embedding dicts from EmbeddingManager
            progress_interval: log progress every N pairs
        
        Returns:
            DataFrame with s1_id, s23_id, label (if available), and 26 features
        """
        # Build row lookup dicts only for entities actually needed in candidate pairs
        # Using itertuples to avoid high memory overhead of 10M pandas Series objects
        s1_needed = set(candidates.keys())
        s23_needed = set()
        for cand_ids in candidates.values():
            s23_needed.update(cand_ids)
            
        s1_cols = [c for c in ['entity_id', 'norm_business_name', 'norm_business_address', 'country'] if c in s1_df.columns]
        s1_lookup = {}
        for row in s1_df[s1_cols].itertuples(index=False):
            eid = getattr(row, 'entity_id')
            if eid in s1_needed:
                s1_lookup[eid] = {
                    'entity_id': eid,
                    'norm_business_name': getattr(row, 'norm_business_name', ''),
                    'norm_business_address': getattr(row, 'norm_business_address', ''),
                    'country': getattr(row, 'country', '')
                }
                
        s23_cols = [c for c in ['entity_id', 'norm_business_name', 'norm_business_address', 'country'] if c in s23_df.columns]
        s23_lookup = {}
        for row in s23_df[s23_cols].itertuples(index=False):
            eid = getattr(row, 'entity_id')
            if eid in s23_needed:
                s23_lookup[eid] = {
                    'entity_id': eid,
                    'norm_business_name': getattr(row, 'norm_business_name', ''),
                    'norm_business_address': getattr(row, 'norm_business_address', ''),
                    'country': getattr(row, 'country', '')
                }
        
        rows = []
        total_pairs = sum(len(v) for v in candidates.values())
        print(f"  Extracting features for {total_pairs} candidate pairs...")
        
        pair_count = 0
        for s1_id, cand_ids in candidates.items():
            s1_row = s1_lookup.get(s1_id)
            if s1_row is None:
                continue
            for s23_id in cand_ids:
                s23_row = s23_lookup.get(s23_id)
                if s23_row is None:
                    continue
                feats = self.extract_pair_features(
                    s1_row, s23_row, name_embs, addr_embs
                )
                feats['s1_id'] = s1_id
                feats['s23_id'] = s23_id
                rows.append(feats)
                
                pair_count += 1
                if pair_count % progress_interval == 0:
                    print(f"    {pair_count}/{total_pairs} pairs processed...")
        
        del s1_lookup, s23_lookup, s1_needed, s23_needed
        import gc
        gc.collect()
        
        print(f"  Feature extraction complete: {len(rows)} pairs")
        return pd.DataFrame(rows)
