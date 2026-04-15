# How to Run This Project

---

## Project Structure

Current repository layout:

```text
DSA4264-text-group4/
├── .env.example
├── README.md
├── requirements.txt
├── setup.sh
├── config/                              # Pipeline and taxonomy configuration
├── data/                                # Local datasets (gitignored)
├── notebooks/                           # Exploration and ad hoc analysis
├── outputs/                             # Generated pipeline outputs
├── reports/                             # Generated markdown reports
├── scripts/                             # Convenience entrypoints
├── tests/                               # Regression tests
└── src/
    ├── ingestion/                       # Standalone scraping utilities
    └── module_readiness/                # Main package
        ├── analysis/                    # Scoring and aggregation
        ├── api/                         # Query surface over pipeline state
        ├── config/                      # Runtime configuration and file helpers
        ├── ingestion/                   # Job and module ingestion
        ├── orchestration/               # Pipeline entrypoint and state container
        ├── processing/                  # Role, skill, and module normalization
        ├── reporting/                   # Markdown report generation
        └── retrieval/                   # BM25, embeddings, and hybrid retrieval
```

---

## Setup

### 1. Create the virtual environment and install dependencies

```bash
bash setup.sh
```

This creates `.venv/`, installs all packages from `requirements.txt`, and creates the `data/` subdirectory structure.
It also registers the repository `src/` directory in the virtual environment's `site-packages`, so imports work without manually exporting `PYTHONPATH`.

Activate the environment manually when working outside of the scripts:

```bash
source .venv/bin/activate
```

If you already have the virtual environment and just want to install the import-path helper, run:

```bash
.venv/bin/python scripts/install_src_path.py
```

### 2. Populate `.env`

Copy the example file and fill in any required values:

```bash
cp .env.example .env
```

E.g open `.env` and set your keys:

```
DATABASE_URL= your_database_url_here
```

---

## Data Setup

Data sources are gitignored and can be accessed by the following sources. Read DATA_SETUP.md for more details on how to access them.

1. MyCareersFuture(MSF) job data
2. NUSMods module data
3. SkillsFuture skills taxonomy

---

## Streamlit Dashboard

The repository now includes a Streamlit dashboard for the Ministry of Education use case:

```bash
streamlit run streamlit_dashboard.py
```

The app is now a multipage Streamlit app:
- main page: curriculum readiness dashboard
- separate page: natural-language job assistant with LLM-backed explanations

The current implementation reads the generated pipeline outputs from `outputs/` and does not rerun the pipeline automatically. Make sure the pipeline has already been run so these tables/files exist:

- `degree_summary`
- `degree_module_map`
- `degree_skill_supply`
- `degree_role_scores`
- `degree_ssoc5_scores`
- `degree_role_skill_gaps`
- `degree_ssoc5_skill_gaps`

The pipeline still writes CSV snapshots to `outputs/` for inspection, but the app now reads from the database so the future web app can use the same source of truth.

If those tables are missing, run the pipeline first:

```bash
.venv/bin/python scripts/run_test2_pipeline.py
```

Optional for the Natural-Language Job Assistant page:
- set `LLM_API_KEY` or `OPENAI_API_KEY` in `.env`
- optionally override `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_TIMEOUT_SECONDS`

---

## Retrieval Evaluation

The repository also includes a retrieval-evaluation script for manually validating the module-to-job ranker.

1. Export a pooled candidate set for labeling:

```bash
python3 scripts/evaluate_retrieval.py export-pool --sample-size 100 --output outputs/retrieval_label_pool.csv
```

This exports exactly 100 modules using faculty-stratified sampling. For each exported module, the sheet keeps the top 10 pooled jobs returned across the `hybrid`, `bm25`, and `embedding` rankers. Fill in the `relevance` column manually with graded labels such as `0`, `1`, `2`, or `3`.

2. Evaluate the labeled file:

```bash
python3 scripts/evaluate_retrieval.py evaluate --labels outputs/retrieval_label_pool.csv --k 10
```

This reports:

- `nDCG@10`
- `Precision@10`
- `Recall@10`

for the `hybrid`, `bm25`, and `embedding` rankers, using the same labeled pool.
