"""Central configuration — loads config.yaml and resolves paths."""
import os, random, yaml, pathlib, numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
STUDENT_ROOT = PROJECT_ROOT.parent.parent  # student_resource/

def _load_config():
    cfg_path = PROJECT_ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
    else:
        raw = {}
    
    # Defaults
    defaults = {
        'seed': 42,
        'device': 'auto',
        'paths': {
            'train_dir': str(STUDENT_ROOT / 'dataset' / 'train'),
            'test_dir': str(STUDENT_ROOT / 'dataset' / 'test'),
            'output_dir': str(STUDENT_ROOT / 'output'),
            'artifacts_dir': str(PROJECT_ROOT / 'artifacts'),
            'model_cache': str(PROJECT_ROOT / 'artifacts' / 'model_cache'),
        },
        'blocking': {
            'max_block_size': 2500,
            'top_k_per_entity': 100,
        },
        'model': {
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': -1,
            'scale_pos_weight': 1.0,
            'early_stopping_rounds': 50,
            'min_child_samples': 20,
        },
        'features': {
            'tfidf_max_features': 100000,
            'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'embedding_dim': 384,
            'embedding_batch_size': 128,
        },
        'threshold': {
            'min_tau': 0.3,
            'max_tau': 0.99,
            'step': 0.01,
        },
        'training': {
            'neg_sample_ratio': 10,
            'val_size': 0.15,
            'val_min_entities': 5000,
        },
    }
    
    # Deep merge raw over defaults
    def merge(base, override):
        result = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = merge(result[k], v)
            else:
                result[k] = v
        return result
    
    cfg = merge(defaults, raw)
    
    # Override paths from env vars
    if os.environ.get('TRAIN_DIR'):
        cfg['paths']['train_dir'] = os.environ['TRAIN_DIR']
    if os.environ.get('TEST_DIR'):
        cfg['paths']['test_dir'] = os.environ['TEST_DIR']
    if os.environ.get('OUTPUT_DIR'):
        cfg['paths']['output_dir'] = os.environ['OUTPUT_DIR']
    if os.environ.get('EMBEDDING_MODEL'):
        cfg['features']['embedding_model'] = os.environ['EMBEDDING_MODEL']
    
    # Ensure directories exist
    for key in ['output_dir', 'artifacts_dir', 'model_cache']:
        os.makedirs(cfg['paths'][key], exist_ok=True)
    
    return cfg

def _detect_device():
    """Auto-detect best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

cfg = _load_config()
if cfg['device'] == 'auto':
    cfg['device'] = _detect_device()
_set_seed(cfg['seed'])

# Convenience aliases
TRAIN_DIR = cfg['paths']['train_dir']
TEST_DIR = cfg['paths']['test_dir']
OUTPUT_DIR = cfg['paths']['output_dir']
ARTIFACTS_DIR = cfg['paths']['artifacts_dir']
DEVICE = cfg['device']
SEED = cfg['seed']
