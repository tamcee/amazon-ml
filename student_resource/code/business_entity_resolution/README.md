# Business Entity Resolution Pipeline

ML pipeline for matching business entities across three independent data sources.

## Setup

```bash
# Clone and install
git clone <repo-url>
cd business_entity_resolution
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Data Setup

Place the dataset files in the expected directory structure:

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/            # generated
├── utils/
│   └── validate_submission.py
└── code/
    └── business_entity_resolution/   # this repo
```

Or override paths with environment variables:

```bash
export TRAIN_DIR=/path/to/train
export TEST_DIR=/path/to/test
export OUTPUT_DIR=/path/to/output
```

## Quick Start

### Full Pipeline (train + infer)

```bash
python run_pipeline.py --stage all
```

### Individual Stages

```bash
# Create dev cohort for fast iteration
python run_pipeline.py --stage dev_cohort --dev-size 1000

# Train only (on dev cohort or full data)
python run_pipeline.py --stage train

# Inference only (requires trained model)
python run_pipeline.py --stage infer
```

### Using a Dev Cohort

```bash
# Create a 1000-entity dev cohort and train on it
python run_pipeline.py --stage dev_cohort --dev-size 1000
export TRAIN_DIR=artifacts/dev_cohort
python run_pipeline.py --stage train
```

## Pipeline Stages

1. **Normalization** — NFKD, Indic transliteration, junk/stutter removal, legal suffix handling
2. **Blocking** — Multi-strategy inverted index (8 key types) with max-block-size guard
3. **Embeddings** — Lazy float16 multilingual embeddings (only candidate-pair entities)
4. **Feature Extraction** — 26 pairwise features (string similarity, TF-IDF, embeddings, phonetic)
5. **Training** — LightGBM with grouped split, negative subsampling, full unsampled validation
6. **Threshold Optimization** — Sweep τ for macro F0.5
7. **Inference** — Score test candidates, apply threshold, write output
8. **Validation** — Check output format with official validator

## Configuration

Edit `config.yaml` or set environment variables. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `blocking.max_block_size` | 2500 | Drop blocking keys exceeding this size |
| `blocking.top_k_per_entity` | 100 | Max candidates per S1 entity |
| `model.n_estimators` | 2000 | Max boosting rounds |
| `training.neg_sample_ratio` | 10 | Negatives per positive in training |
| `features.tfidf_max_features` | 100000 | Max TF-IDF vocabulary size |

## Evaluation Metric

F₀.₅ (precision-heavy): macro-averaged across all S1 entities, including singletons.

## Model

- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (Apache 2.0, 118M params)
- **Classifier:** LightGBM (MIT license)

## Output Files

- `output/matching_results.tsv` — final entity matches (scored on leaderboard)
- `output/candidate_pairs.tsv` — blocking candidate set
