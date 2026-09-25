"""Text normalization — NFKD, transliteration, junk removal, suffix handling."""
import json, os, re, unicodedata
import pandas as pd

# Indic script ranges for detection
INDIC_RANGES = [
    (0x0900, 0x097F, 'devanagari'),
    (0x0980, 0x09FF, 'bengali'),
    (0x0A80, 0x0AFF, 'gujarati'),
    (0x0B80, 0x0BFF, 'tamil'),
    (0x0C00, 0x0C7F, 'telugu'),
    (0x0C80, 0x0CFF, 'kannada'),
    (0x0D00, 0x0D7F, 'malayalam'),
]

# Lazy-load transliteration
_transliteration_available = None
_sanscript = None
_SCRIPT_MAP = None

def _init_transliteration():
    global _transliteration_available, _sanscript, _SCRIPT_MAP
    if _transliteration_available is not None:
        return
    try:
        from indic_transliteration import sanscript
        _sanscript = sanscript
        _SCRIPT_MAP = {
            'devanagari': sanscript.DEVANAGARI,
            'bengali': sanscript.BENGALI,
            'gujarati': sanscript.GUJARATI,
            'tamil': sanscript.TAMIL,
            'telugu': sanscript.TELUGU,
            'kannada': sanscript.KANNADA,
            'malayalam': sanscript.MALAYALAM,
        }
        _transliteration_available = True
    except ImportError:
        _transliteration_available = False
        print("WARNING: indic_transliteration not installed, skipping transliteration")


def detect_indic_script(text: str) -> str | None:
    """Detect the dominant Indic script in text, or None if none found."""
    counts = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, name in INDIC_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=counts.get)


def transliterate_indic(text: str) -> str:
    """Transliterate Indic script text to ITRANS (Latin approximation)."""
    _init_transliteration()
    if not _transliteration_available:
        return text
    script = detect_indic_script(text)
    if script is None or script not in _SCRIPT_MAP:
        return text
    try:
        result = _sanscript.transliterate(text, _SCRIPT_MAP[script], _sanscript.ITRANS)
        return result
    except Exception:
        return text


# Load legal suffixes
def _load_legal_suffixes():
    path = os.path.join(os.path.dirname(__file__), 'data', 'legal_suffixes.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fallback defaults
    return {
        'global': {
            'pvt': 'private', 'ltd': 'limited', 'llc': '', 'inc': '',
            'corp': 'corporation', 'co': 'company', 'llp': '',
            'plc': '', 'gmbh': '', 'ag': '', 'sa': '', 'sas': '',
            'sarl': '', 'eurl': '', 'sasu': '', 'sci': '', 'snc': '',
        }
    }

LEGAL_SUFFIXES = _load_legal_suffixes()

# Compile suffix patterns
def _build_suffix_pattern():
    all_suffixes = set()
    for group in LEGAL_SUFFIXES.values():
        all_suffixes.update(group.keys())
    if not all_suffixes:
        return None
    # Sort by length descending for greedy matching
    sorted_suffixes = sorted(all_suffixes, key=len, reverse=True)
    escaped = [re.escape(s) for s in sorted_suffixes]
    pattern = r'\b(' + '|'.join(escaped) + r')[.,]?\b'
    return re.compile(pattern, re.IGNORECASE)

_SUFFIX_RE = _build_suffix_pattern()

# Junk patterns
_JUNK_PREFIXES = re.compile(r'^[<\-*#@!~|/\\]+\s*')
_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_MARKDOWN_PATTERN = re.compile(r'[\[\](){}*_~`#>|]+')
_MULTI_SPACE = re.compile(r'\s+')
_HASH_NUMBERS = re.compile(r'#\d+\b')


def _normalize_suffix(text: str) -> str:
    """Normalize legal suffixes to canonical forms."""
    if _SUFFIX_RE is None:
        return text
    def replace_suffix(m):
        matched = m.group(1).lower().rstrip('.,').strip()
        # Look up in all groups
        for group in LEGAL_SUFFIXES.values():
            if matched in group:
                replacement = group[matched]
                return replacement if replacement else ''
        return m.group(0)
    return _SUFFIX_RE.sub(replace_suffix, text)


def _collapse_stutters(text: str) -> str:
    """Remove consecutive duplicate tokens: 'the the the' -> 'the'."""
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    deduped = [tokens[0]]
    for t in tokens[1:]:
        if t.lower() != deduped[-1].lower():
            deduped.append(t)
    return ' '.join(deduped)


def normalize_text(text: str) -> str:
    """Apply full normalization pipeline to a single text string."""
    if not isinstance(text, str) or not text.strip():
        return ''
    
    # 1. Unicode NFKD
    text = unicodedata.normalize('NFKD', text)
    
    # 2. Transliterate Indic scripts
    if detect_indic_script(text):
        text = transliterate_indic(text)
    
    # 3. Strip URLs and markdown
    text = _URL_PATTERN.sub(' ', text)
    text = _MARKDOWN_PATTERN.sub(' ', text)
    
    # 4. Strip junk prefixes
    text = _JUNK_PREFIXES.sub('', text)
    
    # 5. Remove hash numbers (e.g. #12345)
    text = _HASH_NUMBERS.sub('', text)
    
    # 6. Lowercase
    text = text.lower()
    
    # 7. Normalize legal suffixes
    text = _normalize_suffix(text)
    
    # 8. Remove remaining punctuation except alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 9. Collapse whitespace
    text = _MULTI_SPACE.sub(' ', text).strip()
    
    # 10. Collapse stutters
    text = _collapse_stutters(text)
    
    return text


def normalize_dataframe(df: pd.DataFrame, name_col='business_name', 
                        addr_col='business_address') -> pd.DataFrame:
    """Normalize name and address columns in-place, adding norm_ prefixed columns."""
    df = df.copy()
    df[f'norm_{name_col}'] = df[name_col].fillna('').apply(normalize_text)
    df[f'norm_{addr_col}'] = df[addr_col].fillna('').apply(normalize_text)
    return df


def normalize_all_sources(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                          cache_dir: str = None) -> tuple:
    """Normalize all three source DataFrames. Optionally cache results."""
    import hashlib
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        
    results = []
    for name, df in [('s1', s1), ('s2', s2), ('s3', s3)]:
        cache_path = os.path.join(cache_dir, f'{name}_normalized.parquet') if cache_dir else None
        
        if cache_path and os.path.exists(cache_path):
            print(f"  Loading cached normalized {name} from {cache_path}")
            results.append(pd.read_parquet(cache_path))
        else:
            print(f"  Normalizing {name} ({len(df)} records)...")
            normed = normalize_dataframe(df)
            if cache_path:
                normed.to_parquet(cache_path, index=False)
                print(f"  Cached to {cache_path}")
            results.append(normed)
    
    return tuple(results)
