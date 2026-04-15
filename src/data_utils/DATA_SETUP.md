# Data Setup — `src/data_utils/`

All pipeline data is stored in a shared Supabase PostgreSQL database and is **not committed to git**.  
This file outlines the data sources, how to load them, and the scripts involved.

## Quick Start

Run `bash src/data_utils/data_setup.sh` from the project root to check what's already in the database. Download data from the following sources and upload to database if it's missing. The script will report the status of each table

---

## Data Sources

### 1. MyCareersFuture (MSF) job postings — `raw_jobs`

**Origin:** Singapore's national job portal ([MyCareersFuture.gov.sg](https://www.mycareersfuture.gov.sg/)). Individual job postings are distributed as one JSON file per posting by the project team via a shared Google Drive folder.

**How to download:**  
**[Download](https://drive.google.com/file/d/17DWvc8xLZz-kT23dEq5BnLeO5aJ1P4t-/view?usp=sharing)** and place into `data/MSF_data/`

---

### 2. NUSMods module catalog — `raw_modules`

**Origin:** [NUSMods public API](https://api.nusmods.com/v2/) — a community-maintained catalog of all NUS modules including titles, descriptions, prerequisites, workload breakdowns, and department/faculty metadata.

**How to download:**  
Run the following commands to run the script `scrape_nusmods.py` to fetch the module data for a given academic year (default: 2024-2025). It caches them as raw JSON under `data/nusmods/<year>/`. Existing cache files are reused automatically; pass `--force-refresh` to re-download everything.

```bash
# Fetch AY2024-2025 (default)
.venv/bin/python src/data_utils/scrape_nusmods.py

# Different year
.venv/bin/python src/data_utils/scrape_nusmods.py --academic-year 2023-2024

# Force re-download
.venv/bin/python src/data_utils/scrape_nusmods.py --force-refresh
```

---

### 3. SkillsFuture skills taxonomy — `skillsfuture_mapping`

**Origin:** Singapore's [SkillsFuture Skills Framework](https://jobsandskills.skillsfuture.gov.sg/frameworks/skills-frameworks), published by SSG.  
Three Excel files must be downloaded manually and placed in `data/`:

| File | Contents |
|---|---|
| `jobsandskills-skillsfuture-skills-framework-dataset.xlsx` | Full TSC/CCS taxonomy with sector, cluster, channel (Technical vs Transferable) |
| `jobsandskills-skillsfuture-unique-skills-list.xlsx` | Deduplicated canonical skill list with descriptions and emerging-skill flags |
| `jobsandskills-skillsfuture-tsc-to-unique-skills-mapping.xlsx` | Mapping from parent skills to sectors for enriching cluster labels |

**How to download:**  
**[Download](https://jobsandskills.skillsfuture.gov.sg/frameworks/skills-frameworks)** the three excels and place them into data directory, then run the following command.

```bash
.venv/bin/python src/data_utils/generate_skillsfuture_mapping.py
```

---

### 4. SSOC 2024 occupation codes — `ssoc2024_definitions`

**Origin:** [Singapore Standard Occupational Classification (SSOC) 2024](https://www.singstat.gov.sg/standard-classifications/national-classifications/singapore-standard-occupational-classification-ssoc), published by the Department of Statistics Singapore.  

**How to download:**  
**[Download](https://www.singstat.gov.sg/standard-classifications/national-classifications/singapore-standard-occupational-classification-ssoc)** the excels and place them into data directory, then run the following command. The script extracts the SSOC codes.

```bash
.venv/bin/python src/data_utils/generate_ssoc_definitions.py
```

---

### 5. NUS degree plan — `nus_degree_plan`

**Origin:** Team-curated curriculum plan CSV for NUS degree requirements and module buckets.

**How to download:**  
Place `nus_degree_plan.csv` into `data/`. The database loader reads it directly and uploads it as the `nus_degree_plan` table.

---

## Scripts Reference

| Script | Purpose | Inputs | Outputs |
|---|---|---|---|
| `scrape_nusmods.py` | Fetch NUSMods module catalog via API | NUSMods API / local cache | `data/nusmods/<year>/` |
| `generate_skillsfuture_mapping.py` | Build normalised skills taxonomy CSV | 3 xlsx in `data/` | `data/skillsfuture_mapping.csv` |
| `generate_ssoc_definitions.py` | Extract SSOC code → title lookup CSV | `data/ssoc2024-detailed-definitions.xlsx` | `data/ssoc2024-detailed-definitions.csv` |
| `db_utils.py` | Load curated NUS degree plan CSV | `data/nus_degree_plan.csv` | Supabase `nus_degree_plan` |
| `db_utils.py` | Load all data into shared database | Raw JSON + processed CSVs | Supabase tables |

---

## Database Tables

| Table | Source script | Contents |
|---|---|---|
| `raw_jobs` | `db_utils.load_raw_jobs` | Flattened MSF job postings |
| `raw_modules` | `db_utils.load_raw_modules` | NUSMods module catalog + detail |
| `nus_degree_plan` | `db_utils.load_nus_degree_plan` | Curated NUS degree curriculum buckets |
| `skillsfuture_mapping` | `generate_skillsfuture_mapping` + `db_utils` | Normalised SkillsFuture skills with channel and cluster |
| `ssoc2024_definitions` | `generate_ssoc_definitions` + `db_utils` | SSOC 2024 code → occupation title lookup |
| `_data_log` | `db_utils` (auto) | Last-loaded timestamp and row count per table |

Pipeline output tables (e.g. `module_summary`, `job_module_scores`) are loaded automatically by `load_pipeline_outputs()` — one table per file in `outputs/`.

---

## Quick Start

```bash
# Check what's in the database and load anything missing
bash src/data_utils/data_setup.sh
```

Add `DATABASE_URL` to your `.env` file — see the **Quick Start** section above for the connection string format.
