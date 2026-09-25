"""Lazy float16 embedding manager — only embeds entities in candidate pairs."""
import os, gc
import numpy as np
import pandas as pd


class EmbeddingManager:
    """Manages sentence embeddings with lazy computation and float16 storage."""
    
    def __init__(self, model_name: str = None, cache_dir: str = None,
                 batch_size: int = 128, device: str = 'cpu'):
        from .config import cfg
        self.model_name = model_name or cfg['features']['embedding_model']
        self.cache_dir = cache_dir or os.path.join(cfg['paths']['artifacts_dir'], 'embeddings')
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._embeddings = {}  # {entity_id: np.array(384,) float16}
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_model(self):
        if self._model is not None:
            return
        
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model: {self.model_name}")
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            # Fallback if offline env vars were set in the environment or if local files were expected
            if 'LocalEntryNotFoundError' in type(e).__name__ or 'offline' in str(e).lower():
                print(f"  Model not found in local cache with offline mode. Retrying with local_files_only=False...")
                self._model = SentenceTransformer(self.model_name, device=self.device, local_files_only=False)
            else:
                raise
    
    def _encode_batch(self, texts: list, batch_size: int = None) -> np.ndarray:
        """Encode texts with OOM backoff on batch size."""
        self._load_model()
        bs = batch_size or self.batch_size
        while bs >= 1:
            try:
                embeddings = self._model.encode(
                    texts, batch_size=bs, show_progress_bar=len(texts) > 1000,
                    normalize_embeddings=True
                )
                return embeddings.astype(np.float16)
            except (RuntimeError, Exception) as e:
                if 'out of memory' in str(e).lower() or 'mps' in str(e).lower():
                    bs = bs // 2
                    print(f"  OOM — reducing batch size to {bs}")
                    gc.collect()
                    if bs < 1:
                        raise
                else:
                    raise
        raise RuntimeError("Batch size reduced to 0")
    
    def _cache_path(self, prefix: str) -> str:
        return os.path.join(self.cache_dir, f'{prefix}_embeddings.npz')
    
    def embed_entities(self, df: pd.DataFrame, entity_ids: set,
                       text_col: str, prefix: str) -> dict:
        """Embed only the specified entity_ids from df.
        
        Args:
            df: DataFrame with entity_id and text_col columns
            entity_ids: set of entity_ids to embed
            text_col: column containing text to embed
            prefix: cache file prefix (e.g. 'name_s2', 'addr_s1')
        
        Returns:
            dict mapping entity_id -> np.array(384,) float16
        """
        cache_path = self._cache_path(prefix)
        
        # Try loading from cache
        if os.path.exists(cache_path):
            print(f"  Loading cached embeddings from {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            result = {str(k): v for k, v in zip(data['ids'], data['embeddings'])}
            # Check if all needed IDs are in cache
            missing = entity_ids - set(result.keys())
            if not missing:
                return {eid: result[eid] for eid in entity_ids if eid in result}
            print(f"  {len(missing)} IDs not in cache, computing...")
        else:
            result = {}
            missing = entity_ids
        
        # Filter to missing IDs
        subset = df[df['entity_id'].isin(missing)].copy()
        if subset.empty:
            return result
        
        texts = subset[text_col].fillna('').tolist()
        ids = subset['entity_id'].tolist()
        
        print(f"  Embedding {len(texts)} texts for {prefix}...")
        embs = self._encode_batch(texts)
        
        for eid, emb in zip(ids, embs):
            result[eid] = emb
        
        # Save cache (all including previously cached)
        all_ids = list(result.keys())
        all_embs = np.array([result[eid] for eid in all_ids], dtype=np.float16)
        np.savez_compressed(cache_path, ids=np.array(all_ids), embeddings=all_embs)
        print(f"  Cached {len(all_ids)} embeddings to {cache_path}")
        
        return {eid: result[eid] for eid in entity_ids if eid in result}
    
    def get_embedding(self, entity_id: str) -> np.ndarray | None:
        """Get a cached embedding by entity_id."""
        return self._embeddings.get(entity_id)
    
    def embed_candidate_pairs(self, s1_df: pd.DataFrame, s23_df: pd.DataFrame,
                               candidates: dict) -> tuple:
        """Embed all entities that appear in candidate pairs.
        
        Returns:
            (name_embeddings, addr_embeddings): dicts of {entity_id: embedding}
        """
        # Collect all entity IDs in candidate pairs
        s1_ids = set(candidates.keys())
        s23_ids = set()
        for cands in candidates.values():
            s23_ids.update(cands)
        
        all_ids = s1_ids | s23_ids
        print(f"  Total unique entities to embed: {len(all_ids)} "
              f"(S1: {len(s1_ids)}, S2/S3: {len(s23_ids)})")
        
        # Combine all dataframes for lookup
        all_df = pd.concat([s1_df, s23_df], ignore_index=True)
        all_df = all_df.drop_duplicates(subset='entity_id')
        
        # Embed names
        name_col = 'norm_business_name' if 'norm_business_name' in all_df.columns else 'business_name'
        addr_col = 'norm_business_address' if 'norm_business_address' in all_df.columns else 'business_address'
        
        print("  Embedding names...")
        name_embs = self.embed_entities(all_df, all_ids, name_col, 'name')
        
        print("  Embedding addresses...")
        addr_embs = self.embed_entities(all_df, all_ids, addr_col, 'addr')
        
        return name_embs, addr_embs
