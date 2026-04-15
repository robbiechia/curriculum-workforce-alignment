from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

from ..config.settings import PipelineConfig
from data_utils.db_utils import read_table


@dataclass
class JobsIngestResult:
    jobs: pd.DataFrame
    diagnostics: Dict[str, float | str]



def _strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    return html.unescape(text)



def _clean_text(raw: str) -> str:
    # Normalize free text aggressively so retrieval compares titles/descriptions on a
    # consistent lexical surface.
    text = _strip_html(raw)
    cleaned_chars: List[str] = []
    for char in text:
        if unicodedata.category(char).startswith("S"):
            continue
        cleaned_chars.append(char)
    text = "".join(cleaned_chars).lower()
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def _clean_skill_text(raw: str) -> str:
    text = _strip_html(raw).lower().strip()
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_skill(skill: str, alias_map: Dict[str, str]) -> str:
    text = _clean_skill_text(skill)
    if not text:
        return ""
    return alias_map.get(text, text)



def _normalize_skills(skills: Iterable[str], alias_map: Dict[str, str]) -> List[str]:
    out = []
    for skill in skills:
        normalized = _normalize_skill(skill, alias_map)
        if normalized:
            out.append(normalized)
    return sorted(set(out))



def _category_names(categories: object) -> List[str]:
    if categories is None:
        return []
    if isinstance(categories, list):
        names = []
        for item in categories:
            if isinstance(item, dict):
                name = item.get("category") or item.get("name")
                if name:
                    names.append(str(name).strip())
            elif isinstance(item, str):
                if item.strip():
                    names.append(item.strip())
        return [c for c in names if c]

    if isinstance(categories, dict):
        name = categories.get("category") or categories.get("name")
        return [str(name).strip()] if name else []

    if isinstance(categories, str) and categories.strip():
        return [categories.strip()]
    return []



def _skill_names(skills: object) -> List[str]:
    if skills is None:
        return []
    if isinstance(skills, list):
        out = []
        for item in skills:
            if isinstance(item, dict):
                name = item.get("skill") or item.get("name")
                if name:
                    out.append(str(name).strip())
            elif isinstance(item, str):
                if item.strip():
                    out.append(item.strip())
        return out
    if isinstance(skills, str) and skills.strip():
        return [skills.strip()]
    return []



def load_jobs(config: PipelineConfig, skill_aliases: Dict[str, str]) -> JobsIngestResult:
    raw = read_table("raw_jobs")
    rows = []
    for _, job in raw.iterrows():
        # The DB stores nested MSF fields as JSON strings, so parse them back before
        # normalizing skills and categories.
        skills_raw = json.loads(job["skills"]) if isinstance(job["skills"], str) else (job["skills"] or [])
        categories_raw = json.loads(job["categories"]) if isinstance(job["categories"], str) else (job["categories"] or [])

        title = str(job.get("title") or "")
        description_raw = str(job.get("description") or "")

        skills = _normalize_skills(skills_raw, skill_aliases)
        categories = [c for c in categories_raw if c]

        title_clean = _clean_text(title)
        description_clean = _clean_text(description_raw)

        try:
            exp = int(float(job["min_experience_years"])) if job["min_experience_years"] is not None else None
        except (TypeError, ValueError):
            exp = None

        rows.append({
            "job_id": str(job["job_id"] or ""),
            "title": title,
            "title_clean": title_clean,
            "description_raw": description_raw,
            "description_clean": description_clean,
            "job_text": f"{title_clean} {description_clean}".strip(),
            "experience_years": exp,
            "ssec_eqa": str(job["ssec_eqa"] or "").strip(),
            "ssoc_code": str(job["ssoc_code"] or "").strip(),
            "skills": skills,
            "categories": categories,
            "primary_category": categories[0] if categories else "Unknown",
            "categories_joined": " | ".join(categories),
            "source_code": None,
            "company": str(job["company_name"] or "").strip(),
            "salary_min": job["salary_min"],
            "salary_max": job["salary_max"],
            "salary_type": str(job["salary_type"] or "").strip(),
            "applications": 0.0,
            "views": 0.0,
            "posted_date": job["posted_at"],
            "expiry_date": job["deleted_at"],
            "raw_path": "",
        })

    df = pd.DataFrame(rows)
    raw_count = len(df)

    if df.empty:
        diagnostics = {
            "jobs_raw_count": 0.0,
            "jobs_after_filters": 0.0,
            "jobs_filter_exp_max": float(config.exp_max),
            "jobs_filter_ssec_eqa": config.primary_ssec_eqa,
        }
        return JobsIngestResult(jobs=df, diagnostics=diagnostics)

    # Restrict the corpus to entry-level roles in the target education band so the
    # downstream module and degree recommendations stay policy-relevant.
    df = df[df["experience_years"].notna()].copy()
    df = df[df["experience_years"] <= int(config.exp_max)].copy()
    df = df[df["ssec_eqa"] == str(config.primary_ssec_eqa)].copy()

    diagnostics: Dict[str, float | str] = {
        "jobs_raw_count": float(raw_count),
        "jobs_after_filters": float(len(df)),
        "jobs_filter_exp_max": float(config.exp_max),
        "jobs_filter_ssec_eqa": str(config.primary_ssec_eqa),
        "jobs_unique_ids": float(df["job_id"].nunique()),
        "jobs_unique_ssoc": float(df["ssoc_code"].nunique()),
        "jobs_unique_categories": float(df["primary_category"].nunique()),
    }

    return JobsIngestResult(jobs=df.reset_index(drop=True), diagnostics=diagnostics)
