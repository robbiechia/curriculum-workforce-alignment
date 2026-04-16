# NUS Curriculum-Workforce Readiness Alignment — DSA4264 Group 4

This project seeks to answer the question: **how well does NUS degree programmes prepare graduates for the Singapore job market?**

The pipeline ingests MyCareersFuture job postings and NUS module data, runs hybrid BM25 + embedding retrieval across 750,000 module-job pairs, and produces role alignment scores for every module and degree. Results are surfaced through a Streamlit dashboard aimed at MOE policy officers.

---

## How to run our pipeline and dashboard

Start by cloning the repo and ensure you are in the project root directory.

```bash
git clone <repo_url>
cd DSA4264-text-group-4
```

Set up your .env file with required credentials by copying the example file and adding your Supabase PostgreSQL connection string and OpenAI API key.

```bash
cp .env.example .env
```

Then run the following commands to set up the environment, run the pipeline, and launch the dashboard:

```bash
bash setup.sh
.venv/bin/python scripts/run_test2_pipeline.py
streamlit run streamlit_dashboard.py
```

---

## Repository layout

```text
DSA4264-text-group-4/
├── .env.example                  # Template — copy to .env and fill in credentials
├── requirements.txt
├── setup.sh                      # Creates .venv, installs dependencies
├── streamlit_dashboard.py        # Dashboard entry point
├── pages/
│   └── 1_Career Query Assistant.py   # Natural-language job assistant (Streamlit page)
├── config/
│   ├── pipeline_config.yaml      # Retrieval and scoring hyperparameters
│   ├── role_clusters.yaml        # 22 role families mapped to 8 broad families
│   ├── role_family_rules.yaml    # Keyword split rules for SSOC role assignment
│   └── skill_aliases.yaml        # Skill normalisation overrides
├── src/
│   ├── data_utils/               # Data ingestion utilities and DB helpers
│   │   ├── DATA_SETUP.md         # Data setup guide (read this before running the pipeline)
│   │   ├── db_utils.py           # PostgreSQL read/write helpers
│   │   ├── scrape_nusmods.py     # Fetch NUSMods catalog via API
│   │   ├── generate_skillsfuture_mapping.py
│   │   └── generate_ssoc_definitions.py
│   └── module_readiness/         # Main package
│       ├── analysis/             # Scoring and aggregation
│       ├── api/                  # Query surface over pipeline state
│       ├── config/               # PipelineConfig and file helpers
│       ├── ingestion/            # Job and module ingestion from DB
│       ├── llm/                  # LLM client and fallback explainer
│       ├── orchestration/        # Pipeline entrypoint and state container
│       ├── processing/           # Role, skill, and module normalisation
│       ├── reporting/            # Markdown report generation
│       └── retrieval/            # BM25, embeddings, hybrid retrieval
├── scripts/
│   ├── run_test2_pipeline.py     # Run the full pipeline end-to-end
│   ├── run_test2_queries.py      # Demo the query API against a completed run
│   ├── evaluate_retrieval.py     # Export and score a retrieval label pool
│   └── install_src_path.py       # Registers src/ on the venv Python path
├── notebooks/
│   ├── eda.ipynb                 # Exploratory data analysis
│   └── report_visualisations.ipynb  # Charts for the technical report
├── outputs/                      # Pipeline output CSVs (gitignored)
├── cache/                        # SHA-256 keyed embedding cache (gitignored)
├── reports/                      # Auto-generated markdown reports
├── docs/
│   └── current/                  # Architecture, runbook, ADRs, glossary
└── tests/                        # Regression and unit tests
```

---

## Our Data Setup

All pipeline data lives in a shared Supabase PostgreSQL database and is not committed to git. If the following tables are missing from the database, **Read [`src/data_utils/DATA_SETUP.md`](src/data_utils/DATA_SETUP.md) for step-by-step download and upload instructions.**

To check if the tables are present, it should be reflected upon the `setup.sh` script output or by running:

```bash
bash src/data_utils/data_setup.sh
```

| Table | Source |
|---|---|
| `raw_jobs` | MCF job postings (JSON files from shared Drive) |
| `raw_modules` | NUSMods API |
| `nus_degree_plan` | Team-curated CSV |
| `skillsfuture_mapping` | SkillsFuture Excel files |
| `ssoc2024_definitions` | SSOC 2024 Excel from SingStat |

---

## 3. Our Module-Job Alignment Pipeline

This runs all stages — ingestion, role assignment, skill taxonomy, module consolidation, retrieval artifact construction, scoring, aggregation, and degree outputs — and writes CSVs to `outputs/`. Embeddings are cached under `cache/embeddings/` so re-runs only recompute new entries.

A full run takes 10–20 minutes on a laptop CPU; the `--quick` flag brings this down to 2–3 minutes by considering a smaller subset of modules.

## 4. Our Streamlit dashboard

The dashboard has two pages:

- **Curriculum Readiness** (main page): Module and degree alignment analysis, skill gap view, role distribution charts.
- **Career Query Assistant** (`pages/1_Career Query Assistant.py`): Natural-language job search — type a role description and get matching early-career jobs and relevant NUS modules.

The dashboard reads directly from the `outputs/` CSVs. Run the pipeline first if any of those files are missing.