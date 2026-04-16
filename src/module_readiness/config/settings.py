from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class PipelineConfig:
    """Runtime configuration for the module readiness pipeline.

    Fields are grouped into four concerns:

    **Job corpus filters** — ``exp_max`` and ``primary_ssec_eqa`` narrow the job
    pool to entry-level graduates so module alignment scores stay relevant to the
    policy question of university preparedness.

    **Retrieval tuning** — BM25 (``bm25_k1``, ``bm25_b``, thresholds) and
    embedding (``embedding_min_similarity``, ``embedding_relative_min``) params
    control which jobs are fused via RRF. ``rrf_k`` and ``retrieval_top_n`` set
    the fusion depth. ``role_support_prior`` damps scores backed by very few jobs.

    **Embedding setup** — model name, batch size, and a dedicated cache directory.
    Embeddings are SHA-256 keyed on text + namespace, so re-runs skip re-encoding.

    **File and directory paths** — all stored as relative ``Path`` objects that
    ``resolve()`` expands to absolute paths against ``project_root``.  Call
    ``from_file()`` rather than constructing this directly; it handles the
    path resolution and YAML merge automatically.
    """

    # These defaults are intentionally centralized here so both scripts and tests can
    # construct a complete config object without hand-populating every field.
    project_root: Path = Path(__file__).resolve().parents[3]
    config_dir: Path = Path("config")

    cache_dir: Path = Path("cache")
    output_dir: Path = Path("outputs")
    reports_dir: Path = Path("reports")

    exp_max: int = 2
    primary_ssec_eqa: str = "70"
    top_k: int = 50

    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    retrieval_top_n: int = 200
    bm25_min_score: float = 20
    bm25_relative_min: float = 0.25
    embedding_min_similarity: float = 0
    embedding_relative_min: float = 0
    role_support_prior: float = 5.0
    degree_role_top_n: int = 5
    degree_demand_skill_top_n: int = 15
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_cache_dir: Path = Path("embeddings")

    # NUSMods ingestion controls
    nusmods_academic_year_candidates: List[str] = field(
        default_factory=lambda: ["2024-2025"]
    )
    nusmods_timeout_seconds: int = 20
    nusmods_fetch_workers: int = 12
    nusmods_max_modules: Optional[int] = None
    nusmods_use_module_details: bool = True

    pipeline_config_file: Path = Path("config/pipeline_config.yaml")
    role_rules_file: Path = Path("config/role_family_rules.yaml")
    role_clusters_file: Path = Path("config/role_clusters.yaml")
    skill_aliases_file: Path = Path("config/skill_aliases.yaml")
    degree_plan_file: Path = Path("data/nus_degree_plan.csv")
    degree_mapping_file: Path = Path("degree_mapping/degree_mapping_AY2425.csv")
    persist_degree_outputs_to_db: bool = False

    def resolve(self) -> "PipelineConfig":
        """Expand all relative paths to absolute ones and create output directories.

        Call this once after construction. ``from_file()`` does it automatically,
        so manual calls are only needed when constructing ``PipelineConfig()`` directly.
        """
        # Resolve all repo-relative paths once up front so downstream modules do not
        # have to reason about the caller's current working directory.
        self.config_dir = (self.project_root / self.config_dir).resolve()
        self.cache_dir = (self.project_root / self.cache_dir).resolve()
        self.output_dir = (self.project_root / self.output_dir).resolve()
        self.reports_dir = (self.project_root / self.reports_dir).resolve()
        self.embedding_cache_dir = Path(self.embedding_cache_dir)

        self.pipeline_config_file = (self.project_root / self.pipeline_config_file).resolve()
        self.role_rules_file = (self.project_root / self.role_rules_file).resolve()
        self.role_clusters_file = (self.project_root / self.role_clusters_file).resolve()
        self.skill_aliases_file = (self.project_root / self.skill_aliases_file).resolve()
        self.degree_plan_file = (self.project_root / self.degree_plan_file).resolve()
        self.degree_mapping_file = (self.project_root / self.degree_mapping_file).resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def from_file(cls, config_path: Path | None = None) -> "PipelineConfig":
        """Load config from a YAML file, falling back to defaults for any missing keys.

        Starts from the dataclass defaults, resolves paths, then overlays any keys
        present in the YAML file.  Paths are re-resolved after the merge so values
        set in YAML are also expanded correctly.  If the config file doesn't exist,
        the resolved defaults are returned as-is — no error is raised.

        Args:
            config_path: Path to the pipeline config YAML.  Defaults to
                         ``config/pipeline_config.yaml`` under ``project_root``.
        """
        base = cls()
        base.resolve()

        final_path = config_path or base.pipeline_config_file
        if not final_path.exists():
            return base

        payload = read_yaml_json(final_path)
        merged = cls(**{**base.__dict__, **payload})
        return merged.resolve()



def read_yaml_json(path: Path) -> Dict[str, Any]:
    """Read JSON-compatible YAML files.

    These config files use JSON syntax and `.yaml` extension to avoid external YAML dependencies.
    """
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)



def write_dataframe_with_fallback(df: pd.DataFrame, parquet_path: Path) -> Tuple[Path, bool]:
    """Write parquet when possible; otherwise fallback to CSV.

    Returns:
        (written_path, parquet_written)
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path, True
    except Exception:
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path, False



def load_json_file(path: Path) -> Dict[str, Any]:
    """Read a JSON file and return its contents as a dict.

    Returns an empty dict if the file doesn't exist, rather than raising.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))



def dump_json_file(path: Path, payload: Dict[str, Any]) -> None:
    """Write a dict to a JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
