#!/usr/bin/env bash
# src/data_utils/data_setup.sh — Check database tables and load any that are missing.
# Always runs relative to the project root (two levels up from this file).
# Usage: bash src/data_utils/data_setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC2046
  export $(grep -v '^\s*#' .env | grep -v '^\s*$' | xargs) 2>/dev/null || true
  set +a
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "ERROR: DATABASE_URL is not set. Add it to .env — see src/data_utils/DATA_SETUP.md."
  exit 1
fi

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

# ---------------------------------------------------------------------------
# Single Python call: check all tables and emit KEY=VALUE lines
# Format: TABLE_STATUS_<name>=yes|no
#         TABLE_INFO_<name>=<last-loaded string or empty>
# ---------------------------------------------------------------------------
db_status=$("$PYTHON" - <<'PYEOF'
import os
tables = ["raw_jobs", "raw_modules", "nus_degree_plan", "skillsfuture_mapping", "ssoc2024_definitions"]
try:
    from sqlalchemy import create_engine, text
    engine = create_engine(os.environ["DATABASE_URL"])
    with engine.connect() as conn:
        # Check whether _data_log exists once up front
        log_exists = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='_data_log')"
        )).scalar()

        for tbl in tables:
            exists = conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name=:t)"
            ), {"t": tbl}).scalar()
            print(f"STATUS_{tbl}={'yes' if exists else 'no'}")
            if exists and log_exists:
                row = conn.execute(text(
                    "SELECT 'Last updated: ' || "
                    "TO_CHAR(loaded_at AT TIME ZONE 'Asia/Singapore', 'YYYY-MM-DD HH24:MI') || "
                    "' SGT, ' || row_count || ' rows' "
                    "FROM _data_log WHERE table_name=:t"
                ), {"t": tbl}).fetchone()
                print(f"INFO_{tbl}={row[0] if row else 'present (no load log)'}")
            elif exists:
                print(f"INFO_{tbl}=present (no load log)")
            else:
                print(f"INFO_{tbl}=not loaded")
except Exception as e:
    for tbl in tables:
        print(f"STATUS_{tbl}=error")
        print(f"INFO_{tbl}=DB error: {e}")
PYEOF
)

# Parse output into variables
_get() { echo "$db_status" | grep "^$1=" | cut -d= -f2-; }

status_raw_jobs=$(_get "STATUS_raw_jobs")
status_raw_modules=$(_get "STATUS_raw_modules")
status_degree_plan=$(_get "STATUS_nus_degree_plan")
status_skillsfuture=$(_get "STATUS_skillsfuture_mapping")
status_ssoc=$(_get "STATUS_ssoc2024_definitions")

info_raw_jobs=$(_get "INFO_raw_jobs")
info_raw_modules=$(_get "INFO_raw_modules")
info_degree_plan=$(_get "INFO_nus_degree_plan")
info_skillsfuture=$(_get "INFO_skillsfuture_mapping")
info_ssoc=$(_get "INFO_ssoc2024_definitions")

_label() { [[ "$1" == "yes" ]] && echo "OK" || echo "MISSING"; }

# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Data Setup Check ==="
echo ""
printf "  %-32s  %-8s  %s\n" "Table" "Status" "Details"
printf "  %-32s  %-8s  %s\n" "-----" "------" "-------"
printf "  %-32s  %-8s  %s\n" "raw_jobs"            "$(_label "$status_raw_jobs")"    "$info_raw_jobs"
printf "  %-32s  %-8s  %s\n" "raw_modules"          "$(_label "$status_raw_modules")" "$info_raw_modules"
printf "  %-32s  %-8s  %s\n" "nus_degree_plan"      "$(_label "$status_degree_plan")" "$info_degree_plan"
printf "  %-32s  %-8s  %s\n" "skillsfuture_mapping" "$(_label "$status_skillsfuture")" "$info_skillsfuture"
printf "  %-32s  %-8s  %s\n" "ssoc2024_definitions" "$(_label "$status_ssoc")"        "$info_ssoc"
echo ""

# Ensure data directories exist
mkdir -p data/MSF_data data/nusmods

# ---------------------------------------------------------------------------
# Load anything that is missing
# ---------------------------------------------------------------------------
ALL_OK=true

# ---- raw_jobs -------------------------------------------------------
if [[ "$status_raw_jobs" != "yes" ]]; then
  ALL_OK=false
  if compgen -G "data/MSF_data/*.json" > /dev/null 2>&1; then
    echo "[raw_jobs] Loading from data/MSF_data ..."
    "$PYTHON" -c "
import sys; from pathlib import Path
sys.path.insert(0, str(Path('src')))
from data_utils.db_utils import get_engine, load_raw_jobs
load_raw_jobs(get_engine())
"
  else
    echo "[raw_jobs] MISSING — download MSF data to data/MSF_data/ first."
    echo "           See: src/data_utils/DATA_SETUP.md"
  fi
fi

# ---- raw_modules ----------------------------------------------------
if [[ "$status_raw_modules" != "yes" ]]; then
  ALL_OK=false
  if [[ -d "data/nusmods" ]] && [[ -n "$(ls -A data/nusmods 2>/dev/null)" ]]; then
    echo "[raw_modules] Loading from data/nusmods ..."
    "$PYTHON" -c "
import sys; from pathlib import Path
sys.path.insert(0, str(Path('src')))
from data_utils.db_utils import get_engine, load_raw_modules
load_raw_modules(get_engine())
"
  else
    echo "[raw_modules] Cache not found — scraping from NUSMods API ..."
    "$PYTHON" src/data_utils/scrape_nusmods.py && \
    "$PYTHON" -c "
import sys; from pathlib import Path
sys.path.insert(0, str(Path('src')))
from data_utils.db_utils import get_engine, load_raw_modules
load_raw_modules(get_engine())
"
  fi
fi

# ---- degree_plan ------------------------------------------
if [[ "$status_degree_plan" != "yes" ]]; then
  ALL_OK=false
  if [[ -f "data/nus_degree_plan.csv" ]]; then
    echo "[nus_degree_plan] Loading from data/nus_degree_plan.csv ..."
    "$PYTHON" -c "
import sys; from pathlib import Path
sys.path.insert(0, str(Path('src')))
from data_utils.db_utils import get_engine, load_nus_degree_plan
load_nus_degree_plan(get_engine())
"
  else
    echo "[nus_degree_plan] MISSING — add data/nus_degree_plan.csv first."
    echo "           See: src/data_utils/DATA_SETUP.md"
  fi
fi

# ---- skillsfuture_mapping ------------------------------------------
if [[ "$status_skillsfuture" != "yes" ]]; then
  ALL_OK=false
  sf1="data/jobsandskills-skillsfuture-skills-framework-dataset.xlsx"
  sf2="data/jobsandskills-skillsfuture-unique-skills-list.xlsx"
  sf3="data/jobsandskills-skillsfuture-tsc-to-unique-skills-mapping.xlsx"
  if [[ -f "$sf1" && -f "$sf2" && -f "$sf3" ]]; then
    echo "[skillsfuture_mapping] Generating from Excel files ..."
    "$PYTHON" src/data_utils/generate_skillsfuture_mapping.py || { echo "[error] Generation failed — aborting load."; exit 1; }
    echo "[skillsfuture_mapping] Loading into database ..."
    "$PYTHON" -c "
import sys, pandas as pd; from pathlib import Path
sys.path.insert(0, str(Path('src')))
from data_utils.db_utils import get_engine, write_table, _log_load
engine = get_engine()
df = pd.read_csv('data/skillsfuture_mapping.csv')
write_table(df, 'skillsfuture_mapping', engine=engine)
_log_load('skillsfuture_mapping', len(df), engine)
print(f'  skillsfuture_mapping: {len(df):,} rows')
"
  else
    echo "[skillsfuture_mapping] MISSING — download SkillsFuture Excel files to data/ first."
    echo "           See: src/data_utils/DATA_SETUP.md"
  fi
fi

# ---- ssoc2024_definitions ------------------------------------------
if [[ "$status_ssoc" != "yes" ]]; then
  ALL_OK=false
  if [[ -f "data/ssoc2024-detailed-definitions.xlsx" ]]; then
    echo "[ssoc2024_definitions] Generating from Excel file ..."
    "$PYTHON" src/data_utils/generate_ssoc_definitions.py || { echo "[error] Generation failed — aborting load."; exit 1; }
    echo "[ssoc2024_definitions] Loading into database ..."
    "$PYTHON" -c "
import sys, pandas as pd; from pathlib import Path
sys.path.insert(0, str(Path('src')))
from data_utils.db_utils import get_engine, write_table, _log_load
engine = get_engine()
df = pd.read_csv('data/ssoc2024-detailed-definitions.csv', dtype=str)
write_table(df, 'ssoc2024_definitions', engine=engine)
_log_load('ssoc2024_definitions', len(df), engine)
print(f'  ssoc2024_definitions: {len(df):,} rows')
"
  else
    echo "[ssoc2024_definitions] MISSING — download ssoc2024-detailed-definitions.xlsx to data/ first."
    echo "           See: src/data_utils/DATA_SETUP.md"
  fi
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
if $ALL_OK; then
  echo "All tables present. You are ready to run the pipeline."
else
  echo "Re-run 'bash src/data_utils/data_setup.sh' after resolving any manual steps above."
fi
echo ""
