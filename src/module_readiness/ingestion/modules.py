from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ..config.settings import PipelineConfig


@dataclass
class ModulesIngestResult:
    modules: pd.DataFrame
    diagnostics: Dict[str, float | str]



def _module_prefix(code: str) -> str:
    m = re.match(r"^([A-Z]{2,4})", (code or "").upper())
    return m.group(1) if m else ""


def _module_level(code: str) -> int | None:
    m = re.match(r"^[A-Z]{2,4}(\d)", (code or "").upper())
    if not m:
        return None
    return int(m.group(1))


def _is_undergraduate_module(code: str) -> bool:
    level = _module_level(code)
    return level is not None and 1 <= level <= 4



def _profile_from_prefix(prefix: str, role_rules: Dict[str, object]) -> str:
    mapping = role_rules.get("module_prefix_profiles", {})
    if isinstance(mapping, dict):
        return str(mapping.get(prefix, "general"))
    return "general"



def _seed_terms(profile: str) -> str:
    if profile == "computing":
        return "python data analysis machine learning sql software engineering"
    if profile == "quantitative":
        return "statistics optimization quantitative modelling data methods"
    if profile == "business":
        return "finance accounting management operations marketing"
    if profile == "social_science":
        return "policy analysis research methods social science evidence"
    if profile == "communications":
        return "communications media writing campaigns stakeholder messaging"
    if profile == "engineering":
        return "systems design engineering process control optimization"
    if profile == "health":
        return "healthcare clinical biomedical evidence patient outcomes"
    return "critical thinking problem solving communication"



def _select_modules(
    module_list: List[Dict],
    max_modules: Optional[int],
) -> List[Dict]:
    selected = []

    for item in module_list:
        code = str(item.get("moduleCode") or "").strip().upper()
        if not code:
            continue
        # The curriculum analysis is undergraduate-focused, so only 1K-4K modules are kept.
        if not _is_undergraduate_module(code):
            continue
        selected.append(item)

    selected.sort(key=lambda x: str(x.get("moduleCode") or ""))

    if max_modules is not None and max_modules > 0 and len(selected) > max_modules:
        # Round-robin selection by module prefix to avoid alphabetical/prefix bias.
        pools: Dict[str, List[Dict]] = defaultdict(list)
        for item in selected:
            code = str(item.get("moduleCode") or "").strip().upper()
            pools[_module_prefix(code)].append(item)

        for prefix in pools:
            pools[prefix].sort(key=lambda x: str(x.get("moduleCode") or ""))

        picked: List[Dict] = []
        prefixes = sorted(pools.keys())
        idx_map = {p: 0 for p in prefixes}

        while len(picked) < max_modules:
            progressed = False
            for prefix in prefixes:
                idx = idx_map[prefix]
                if idx >= len(pools[prefix]):
                    continue
                picked.append(pools[prefix][idx])
                idx_map[prefix] += 1
                progressed = True
                if len(picked) >= max_modules:
                    break
            if not progressed:
                break

        selected = picked

    dedup = {}
    for item in selected:
        code = str(item.get("moduleCode") or "").strip().upper()
        if code:
            dedup[code] = item

    return list(sorted(dedup.values(), key=lambda x: str(x.get("moduleCode") or "")))



def _format_workload_token(workload: object) -> str:
    if not isinstance(workload, list):
        return ""
    labels = ["lecture", "tutorial", "lab", "project", "preparation"]
    parts = []
    for idx, value in enumerate(workload[:5]):
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        label = labels[idx] if idx < len(labels) else f"slot_{idx}"
        parts.append(f"{label}_{v:g}")
    return " ".join(parts)



def load_nus_modules(config: PipelineConfig, role_rules: Dict[str, object]) -> ModulesIngestResult:
    from data_utils.db_utils import read_table

    df_raw = read_table("raw_modules")

    # Build module_list and detail_map from DB rows
    module_list: List[Dict] = []
    detail_map: Dict[str, Dict] = {}

    def _db_val(v):
        """Convert pandas NaN / NA to None so downstream `(v or "")` logic works."""
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v

    for _, row in df_raw.iterrows():
        code = str(row.get("moduleCode") or "").strip().upper()
        if not code:
            continue

        # Parse workload back from JSON string to list
        workload_raw = _db_val(row.get("workload"))
        if isinstance(workload_raw, str):
            try:
                workload_parsed = json.loads(workload_raw)
            except (json.JSONDecodeError, ValueError):
                workload_parsed = None
        else:
            workload_parsed = workload_raw

        module_list.append({"moduleCode": code, "title": _db_val(row.get("title")) or code})
        detail_map[code] = {
            "title": _db_val(row.get("title")),
            "description": _db_val(row.get("description")),
            "additionalInformation": _db_val(row.get("additionalInformation")),
            "prerequisite": _db_val(row.get("prerequisite")),
            "preclusion": _db_val(row.get("preclusion")),
            "department": _db_val(row.get("department")),
            "faculty": _db_val(row.get("faculty")),
            "moduleCredit": _db_val(row.get("moduleCredit")),
            "workload": workload_parsed,
        }

    year = "2024-2025"
    if not df_raw.empty and "acadYear" in df_raw.columns:
        raw_year = str(df_raw["acadYear"].dropna().iloc[0]) if df_raw["acadYear"].notna().any() else year
        year = raw_year.replace("/", "-")

    selected = _select_modules(
        module_list=module_list,
        max_modules=config.nusmods_max_modules,
    )

    rows: List[Dict[str, object]] = []
    for item in selected:
        code = str(item.get("moduleCode") or "").strip().upper()
        if not code:
            continue

        prefix = _module_prefix(code)
        profile = _profile_from_prefix(prefix, role_rules)
        seed = _seed_terms(profile)

        detail = detail_map.get(code, {})
        title = (detail.get("title") or item.get("title") or code).strip()

        description = (detail.get("description") or "").strip()
        additional = (detail.get("additionalInformation") or "").strip()
        prerequisite = (detail.get("prerequisite") or "").strip()
        preclusion = (detail.get("preclusion") or "").strip()
        department = (detail.get("department") or "").strip()
        faculty = (detail.get("faculty") or "").strip()
        module_credit = str(detail.get("moduleCredit") or "").strip()
        workload = _format_workload_token(detail.get("workload"))

        if not description:
            description = title

        # `module_text` is the broad retrieval representation. It intentionally includes
        # more context than the narrower text later used for skill extraction.
        module_text_parts = [
            title, description, additional, prerequisite, preclusion,
            department, faculty, module_credit, workload, seed,
        ]
        module_text = " ".join(part for part in module_text_parts if part).strip()

        rows.append({
            "module_code": code,
            "module_title": title,
            "module_prefix": prefix,
            "module_profile": profile,
            "module_description": description,
            "module_workload": workload,
            "module_text": module_text,
            "module_department": department,
            "module_faculty": faculty,
            "module_credit": module_credit,
            "module_source": "db",
            "module_detail_available": 1 if code in detail_map else 0,
            "module_academic_year": year,
        })

    if not rows:
        for code, payload in config.focus_module_overrides.items():
            if not _is_undergraduate_module(code):
                continue
            prefix = _module_prefix(code)
            profile = _profile_from_prefix(prefix, role_rules)
            description = payload.get("description", code)
            workload = payload.get("workload", "")
            module_text = " ".join([code, description, workload, _seed_terms(profile)])
            rows.append({
                "module_code": code,
                "module_title": code,
                "module_prefix": prefix,
                "module_profile": profile,
                "module_description": description,
                "module_workload": workload,
                "module_text": module_text,
                "module_department": "",
                "module_faculty": "",
                "module_credit": "",
                "module_source": "seed",
                "module_detail_available": 0,
                "module_academic_year": year,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["module_code"]).reset_index(drop=True)

    diagnostics: Dict[str, float | str] = {
        "modules_rows": float(len(df)),
        "modules_catalog_year": year,
        "modules_catalog_source": "db",
        "modules_catalog_size": float(len(module_list)),
        "modules_selected_before_detail": float(len(selected)),
        "modules_detail_available_rows": float(df["module_detail_available"].sum()) if not df.empty else 0.0,
        "modules_unique_prefixes": float(df["module_prefix"].nunique()) if not df.empty else 0.0,
    }

    return ModulesIngestResult(modules=df, diagnostics=diagnostics)
