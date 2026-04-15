from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ..config import PipelineConfig
from ..orchestration import ModuleReadinessState
from ..retrieval import HybridRetrievalEngine, build_retrieval_artifacts
from .query import ModuleReadinessQueryAPI


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except pd.errors.ParserError:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh, skipinitialspace=True))
        if not rows:
            return pd.DataFrame()
        header = [str(v).strip() for v in rows[0]]
        body = [
            (([str(v).strip() for v in row]) + [""] * len(header))[: len(header)]
            for row in rows[1:]
        ]
        df = pd.DataFrame(body, columns=header)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda v: str(v).strip())
    return df


def _parse_list(value: object) -> list[str]:
    if not value or str(value).strip() in {"", "nan"}:
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [item.strip().lower() for item in str(value).split(";") if item.strip()]


def _coerce_bool(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    trues = {"true", "1", "yes", "y"}
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: str(v).strip().lower() in trues)
    return out


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truncate(text: object, limit: int = 260) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _join_unique(values: Sequence[object], limit: int = 3, sep: str = "; ") -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return sep.join(out)


@dataclass
class JobQueryRunResult:
    jobs: pd.DataFrame
    modules: pd.DataFrame
    degree_id: str | None = None
    degree_label: str | None = None


@dataclass
class DashboardQueryBackend:
    state: ModuleReadinessState
    api: ModuleReadinessQueryAPI
    module_job_evidence: pd.DataFrame
    degree_module_map: pd.DataFrame
    degree_summary: pd.DataFrame

    def degree_label_for_id(self, degree_id: str | None) -> str | None:
        if not degree_id:
            return None
        subset = self.degree_summary[self.degree_summary["degree_id"].astype(str) == str(degree_id)]
        if subset.empty:
            return None
        return str(subset.iloc[0].get("primary_major", "")).strip() or str(degree_id)

    def required_module_codes_for_degree(self, degree_id: str | None) -> set[str]:
        if not degree_id or self.degree_module_map.empty:
            return set()
        subset = self.degree_module_map[
            (self.degree_module_map["degree_id"].astype(str) == str(degree_id))
            & self.degree_module_map["module_found"].astype(bool)
            & ~self.degree_module_map["is_unrestricted_elective"].astype(bool)
        ]
        return {
            str(code).strip().upper()
            for code in subset.get("module_code", pd.Series(dtype=str)).tolist()
            if str(code).strip()
        }

    def _selected_job_skill_vocab(self, job_ids: Sequence[str]) -> tuple[set[str], set[str]]:
        subset = self.state.jobs[self.state.jobs["job_id"].isin(set(job_ids))].copy()
        tech_vocab: set[str] = set()
        soft_vocab: set[str] = set()
        for skills in subset.get("technical_skills", pd.Series(dtype=object)):
            tech_vocab.update(_parse_list(skills))
        for skills in subset.get("soft_skills", pd.Series(dtype=object)):
            soft_vocab.update(_parse_list(skills))
        return tech_vocab, soft_vocab

    def _enrich_jobs(self, jobs: pd.DataFrame) -> pd.DataFrame:
        if jobs.empty:
            return jobs
        meta_cols = [
            "job_id",
            "description_clean",
            "technical_skills",
            "soft_skills",
            "posted_date",
            "salary_min",
            "salary_max",
            "salary_type",
        ]
        meta_cols = [col for col in meta_cols if col in self.state.jobs.columns]
        meta = self.state.jobs[meta_cols].copy()
        if "technical_skills" in meta.columns:
            meta["technical_skills"] = meta["technical_skills"].apply(_parse_list)
        if "soft_skills" in meta.columns:
            meta["soft_skills"] = meta["soft_skills"].apply(_parse_list)
        out = jobs.merge(meta, on="job_id", how="left")
        if "description_clean" in out.columns:
            out["job_summary"] = out["description_clean"].map(lambda text: _truncate(text, limit=280))
        else:
            out["job_summary"] = ""
        return out

    def _module_match_metadata(self, module_codes: Sequence[str], job_ids: Sequence[str]) -> pd.DataFrame:
        if not module_codes or not job_ids or self.module_job_evidence.empty:
            return pd.DataFrame(
                columns=[
                    "module_code",
                    "matched_job_count",
                    "matched_jobs",
                    "selected_job_avg_rrf_score",
                    "selected_job_max_rrf_score",
                    "job_evidence_terms",
                ]
            )

        subset = self.module_job_evidence[
            self.module_job_evidence["module_code"].astype(str).isin({str(code) for code in module_codes})
            & self.module_job_evidence["job_id"].astype(str).isin({str(job_id) for job_id in job_ids})
        ].copy()
        if subset.empty:
            return pd.DataFrame(
                columns=[
                    "module_code",
                    "matched_job_count",
                    "matched_jobs",
                    "selected_job_avg_rrf_score",
                    "selected_job_max_rrf_score",
                    "job_evidence_terms",
                ]
            )

        subset["rrf_score"] = pd.to_numeric(subset["rrf_score"], errors="coerce").fillna(0.0)
        subset["job_label"] = subset.apply(
            lambda row: f"{row['job_title']} ({row['company']})" if str(row.get("company", "")).strip() else str(row["job_title"]),
            axis=1,
        )
        subset = subset.sort_values(["module_code", "rrf_score"], ascending=[True, False])
        rows: list[dict[str, object]] = []
        for module_code, group in subset.groupby("module_code", sort=False):
            rows.append(
                {
                    "module_code": str(module_code),
                    "matched_job_count": int(group["job_id"].nunique()),
                    "matched_jobs": _join_unique(group["job_label"].tolist(), limit=3),
                    "selected_job_avg_rrf_score": float(group["rrf_score"].mean()),
                    "selected_job_max_rrf_score": float(group["rrf_score"].max()),
                    "job_evidence_terms": _join_unique(group["evidence_terms"].tolist(), limit=3, sep=" | "),
                }
            )
        return pd.DataFrame(rows)

    def _enrich_modules(
        self,
        modules: pd.DataFrame,
        job_ids: Sequence[str],
        degree_id: str | None = None,
    ) -> pd.DataFrame:
        if modules.empty:
            return modules

        meta_cols = [
            "module_code",
            "module_description",
            "module_faculty",
            "module_department",
            "module_credit",
            "technical_skills",
            "soft_skills",
        ]
        meta_cols = [col for col in meta_cols if col in self.state.modules.columns]
        meta = self.state.modules[meta_cols].copy()
        if "technical_skills" in meta.columns:
            meta["technical_skills"] = meta["technical_skills"].apply(_parse_list)
        if "soft_skills" in meta.columns:
            meta["soft_skills"] = meta["soft_skills"].apply(_parse_list)

        out = modules.copy()
        out["module_code"] = out["module_code"].astype(str).str.strip().str.upper()
        out = out.merge(meta, on="module_code", how="left")
        out["module_summary"] = out.get("module_description", pd.Series("", index=out.index)).map(
            lambda text: _truncate(text, limit=280)
        )

        evidence_meta = self._module_match_metadata(out["module_code"].tolist(), job_ids)
        out = out.merge(evidence_meta, on="module_code", how="left")
        out["matched_job_count"] = pd.to_numeric(out.get("matched_job_count"), errors="coerce").fillna(0).astype(int)
        out["selected_job_avg_rrf_score"] = pd.to_numeric(
            out.get("selected_job_avg_rrf_score"),
            errors="coerce",
        ).fillna(0.0)
        out["selected_job_max_rrf_score"] = pd.to_numeric(
            out.get("selected_job_max_rrf_score"),
            errors="coerce",
        ).fillna(0.0)

        tech_vocab, soft_vocab = self._selected_job_skill_vocab(job_ids)
        out["technical_skill_overlap"] = out.get(
            "technical_skills", pd.Series([[] for _ in range(len(out))], index=out.index)
        ).apply(lambda skills: [skill for skill in _parse_list(skills) if skill in tech_vocab][:6])
        out["soft_skill_overlap"] = out.get(
            "soft_skills", pd.Series([[] for _ in range(len(out))], index=out.index)
        ).apply(lambda skills: [skill for skill in _parse_list(skills) if skill in soft_vocab][:4])

        required_codes = self.required_module_codes_for_degree(degree_id)
        out["in_selected_degree"] = out["module_code"].isin(required_codes) if required_codes else False
        return out

    def run_job_query(
        self,
        natural_language_query: str,
        *,
        top_job_k: int = 5,
        top_module_k: int = 8,
        exp_max: int = 2,
        degree_id: str | None = None,
    ) -> JobQueryRunResult:
        jobs = self.api.search_jobs(
            natural_language_query=natural_language_query,
            exp_max=int(exp_max),
            top_k=int(top_job_k),
        )
        if jobs.empty:
            return JobQueryRunResult(
                jobs=jobs,
                modules=pd.DataFrame(),
                degree_id=degree_id,
                degree_label=self.degree_label_for_id(degree_id),
            )

        allowed_module_codes = self.required_module_codes_for_degree(degree_id)
        modules = self.api.recommend_relevant_modules(
            jobs["job_id"].tolist(),
            top_k=int(top_module_k),
            allowed_module_codes=sorted(allowed_module_codes) if allowed_module_codes else None,
        )
        jobs = self._enrich_jobs(jobs)
        modules = self._enrich_modules(modules, jobs["job_id"].tolist(), degree_id=degree_id)
        return JobQueryRunResult(
            jobs=jobs,
            modules=modules,
            degree_id=degree_id,
            degree_label=self.degree_label_for_id(degree_id),
        )


def load_dashboard_query_backend(
    *,
    output_dir: Path | None = None,
    config: PipelineConfig | None = None,
) -> DashboardQueryBackend:
    config = config or PipelineConfig.from_file()
    output_dir = (output_dir or config.output_dir).resolve()

    jobs = _read_csv(output_dir / "jobs_clean.csv")
    modules = _read_csv(output_dir / "modules_clean.csv")
    module_role_scores = _read_csv(output_dir / "module_role_scores.csv")
    module_summary = _read_csv(output_dir / "module_summary.csv")
    module_job_evidence = _read_csv(output_dir / "module_job_evidence.csv")
    degree_module_map = _coerce_bool(
        _read_csv(output_dir / "degree_module_map.csv"),
        ["is_unrestricted_elective", "module_found"],
    )
    degree_summary = _read_csv(output_dir / "degree_summary.csv")

    jobs["experience_years"] = pd.to_numeric(jobs["experience_years"], errors="coerce").fillna(9999)
    for col in ["technical_skills", "soft_skills"]:
        if col in jobs.columns:
            jobs[col] = jobs[col].apply(_parse_list)
        if col in modules.columns:
            modules[col] = modules[col].apply(_parse_list)
    module_role_scores["role_score"] = pd.to_numeric(module_role_scores["role_score"], errors="coerce").fillna(0.0)
    if "evidence_job_count" in module_role_scores.columns:
        module_role_scores["evidence_job_count"] = pd.to_numeric(
            module_role_scores["evidence_job_count"], errors="coerce"
        ).fillna(0.0)
    if "top_role_score" in module_summary.columns:
        module_summary["top_role_score"] = pd.to_numeric(module_summary["top_role_score"], errors="coerce").fillna(0.0)

    retrieval_artifacts = build_retrieval_artifacts(
        config=config,
        jobs=jobs,
        modules=modules,
    )
    retrieval = HybridRetrievalEngine(config=config, artifacts=retrieval_artifacts)

    known_skills = sorted(
        {
            skill
            for frame in (jobs, modules)
            for col in ("technical_skills", "soft_skills")
            if col in frame.columns
            for values in frame[col].tolist()
            for skill in _parse_list(values)
        }
    )

    state = ModuleReadinessState(
        config=config,
        role_rules={},
        jobs=jobs,
        modules=modules,
        module_job_scores=module_job_evidence,
        module_ssoc5_scores=pd.DataFrame(),
        module_role_scores=module_role_scores,
        module_summary=module_summary,
        module_gap_summary=pd.DataFrame(),
        retrieval_artifacts=retrieval_artifacts,
        retrieval=retrieval,
        skill_channel_map={},
        known_skills=known_skills,
        diagnostics={},
        degree_module_map=degree_module_map,
        degree_summary=degree_summary,
    )
    api = ModuleReadinessQueryAPI(state)
    return DashboardQueryBackend(
        state=state,
        api=api,
        module_job_evidence=module_job_evidence,
        degree_module_map=degree_module_map,
        degree_summary=degree_summary,
    )
