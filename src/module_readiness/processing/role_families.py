from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class RoleFamilyResult:
    jobs: pd.DataFrame
    diagnostics: Dict[str, float | str]



def _ssoc_digits(code: str) -> str:
    return "".join(ch for ch in str(code or "").strip() if ch.isdigit())



def _ssoc_prefix(code: str, width: int) -> str:
    digits = _ssoc_digits(code)
    if len(digits) < width:
        return ""
    return digits[:width]



def _keyword_match(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)



def _normalize_title_for_label(title: str) -> str:
    text = str(title or "").lower().strip()
    if not text:
        return ""
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^\)]*\)", " ", text)
    text = re.sub(r"^\s*\d+\s*[-:]\s*", "", text)
    text = re.sub(r"[^a-z0-9\s/&-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def _title_case_label(text: str) -> str:
    return " ".join(token.capitalize() for token in str(text or "").split())


def _row_text_blob(row: pd.Series) -> str:
    skills = row.get("skills", [])
    if not isinstance(skills, list):
        skills = []
    return " ".join(
        part
        for part in [
            str(row.get("title", "")).strip(),
            str(row.get("description_clean", "")).strip(),
            str(row.get("ssoc_5d_name", "")).strip(),
            str(row.get("ssoc_4d_name", "")).strip(),
            str(row.get("primary_category", "")).strip(),
            " ".join(str(skill).strip() for skill in skills if str(skill).strip()),
        ]
        if part
    ).lower()


def _cluster_rule_matches(row: pd.Series, text_blob: str, rule: Dict[str, object]) -> bool:
    ssoc4_filter = {str(code).strip() for code in rule.get("ssoc4", []) if str(code).strip()}
    if ssoc4_filter and str(row.get("ssoc_4d", "")).strip() not in ssoc4_filter:
        return False

    ssoc5_filter = {str(code).strip() for code in rule.get("ssoc5", []) if str(code).strip()}
    if ssoc5_filter and str(row.get("ssoc_5d", "")).strip() not in ssoc5_filter:
        return False

    categories = {str(cat).strip() for cat in rule.get("categories", []) if str(cat).strip()}
    if categories and str(row.get("primary_category", "")).strip() not in categories:
        return False

    keywords = [str(keyword).strip().lower() for keyword in rule.get("keywords", []) if str(keyword).strip()]
    if keywords and not any(keyword in text_blob for keyword in keywords):
        return False

    return True


def _assign_role_cluster(
    row: pd.Series,
    cluster_rules: Dict[str, object],
    legacy_family: str,
) -> Tuple[str, str, str]:
    ssoc5_exact = cluster_rules.get("ssoc5_exact_map", {})
    ssoc4_exact = cluster_rules.get("ssoc4_exact_map", {})
    broad_map = cluster_rules.get("cluster_broad_family_map", {})
    split_rules = cluster_rules.get("split_rules", [])

    ssoc_5d = str(row.get("ssoc_5d", "")).strip()
    ssoc_4d = str(row.get("ssoc_4d", "")).strip()
    text_blob = _row_text_blob(row)

    cluster = ""
    source = ""

    if isinstance(ssoc5_exact, dict) and ssoc_5d in ssoc5_exact:
        cluster = str(ssoc5_exact[ssoc_5d]).strip()
        source = "ssoc5_exact"
    elif isinstance(ssoc4_exact, dict) and ssoc_4d in ssoc4_exact:
        cluster = str(ssoc4_exact[ssoc_4d]).strip()
        source = "ssoc4_exact"
    elif isinstance(split_rules, list):
        for rule in split_rules:
            if not isinstance(rule, dict):
                continue
            if not _cluster_rule_matches(row, text_blob, rule):
                continue
            cluster = str(rule.get("cluster", "")).strip()
            if cluster:
                source = "split_rule"
                break

    if not cluster and legacy_family and not legacy_family.isdigit() and legacy_family != "Other":
        cluster = legacy_family
        source = "legacy_family"

    if not cluster:
        cluster = "Other"
        source = "fallback"

    broad_family = str(broad_map.get(cluster, "Other")).strip() if isinstance(broad_map, dict) else "Other"
    if not broad_family:
        broad_family = "Other"

    return cluster, source, broad_family



def _build_ssoc_name_map(jobs: pd.DataFrame, code_col: str) -> Dict[str, str]:
    if jobs.empty or code_col not in jobs.columns or "title" not in jobs.columns:
        return {}

    tmp = jobs[[code_col, "title"]].copy()
    tmp[code_col] = tmp[code_col].astype(str).str.strip()
    tmp = tmp[tmp[code_col] != ""]
    if tmp.empty:
        return {}

    tmp["label"] = tmp["title"].apply(_normalize_title_for_label)
    tmp = tmp[tmp["label"] != ""]
    if tmp.empty:
        return {}

    counts = (
        tmp.groupby([code_col, "label"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values([code_col, "n", "label"], ascending=[True, False, True])
    )
    top = counts.groupby(code_col, as_index=False).head(1)

    out: Dict[str, str] = {}
    for _, row in top.iterrows():
        code = str(row[code_col]).strip()
        label = _title_case_label(str(row["label"]).strip())
        if code and label:
            out[code] = label
    return out



def _clean_ssoc_title(title: str) -> str:
    text = str(title or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _load_ssoc_title_maps(config=None) -> Tuple[Dict[str, str], Dict[str, str]]:
    from data_utils.db_utils import read_table
    from pathlib import Path as _Path

    try:
        if config is not None and getattr(config, "use_local_data", False):
            df = pd.read_csv(
                _Path(str(config.local_ssoc_csv)), dtype=str, encoding="utf-8-sig", keep_default_na=False
            )
        else:
            df = read_table("ssoc2024_definitions")
        map_4d: Dict[str, str] = {}
        map_5d: Dict[str, str] = {}
        for _, row in df.iterrows():
            code = _ssoc_digits(str(row.get("SSOC 2024") or ""))
            title = _clean_ssoc_title(str(row.get("SSOC 2024 Title") or ""))
            if not code or not title:
                continue
            if len(code) == 4:
                map_4d.setdefault(code, title)
            elif len(code) == 5:
                map_5d.setdefault(code, title)
        return map_4d, map_5d
    except Exception:
        return {}, {}



def assign_role_families(
    jobs: pd.DataFrame,
    role_rules: Dict[str, object],
    cluster_rules: Dict[str, object] | None = None,
) -> RoleFamilyResult:
    if jobs.empty:
        return RoleFamilyResult(jobs=jobs.copy(), diagnostics={"role_family_unique": 0.0})

    ssoc_map = role_rules.get("ssoc_prefix_map", {})
    category_map = role_rules.get("category_rules", {})
    keyword_rules = role_rules.get("title_keyword_rules", [])

    # The fallback order here matters: prefer official SSOC structure first, then use
    # keyword/category rules only when SSOC is unavailable or malformed.
    families: List[str] = []
    family_source: List[str] = []
    ssoc_5d_codes: List[str] = []
    ssoc_4d_codes: List[str] = []

    for _, row in jobs.iterrows():
        ssoc_raw = str(row.get("ssoc_code", ""))
        ssoc_5d = _ssoc_prefix(ssoc_raw, width=5)
        ssoc_4d = _ssoc_prefix(ssoc_raw, width=4)

        # Primary backbone for this iteration:
        # use SSOC-5 for fine-grained evidence and SSOC-4 for final grouping label.
        if ssoc_5d and ssoc_4d:
            families.append(ssoc_4d)
            family_source.append("ssoc4_from_ssoc5")
            ssoc_5d_codes.append(ssoc_5d)
            ssoc_4d_codes.append(ssoc_4d)
            continue

        # Fallback logic for malformed/missing SSOC codes.
        ssoc_2d = _ssoc_prefix(ssoc_raw, width=2)
        title = str(row.get("title", ""))
        category = str(row.get("primary_category", "Unknown"))
        skills = row.get("skills", [])
        if not isinstance(skills, list):
            skills = []

        if isinstance(ssoc_map, dict) and ssoc_2d in ssoc_map:
            families.append(str(ssoc_map[ssoc_2d]))
            family_source.append("ssoc2_map")
            ssoc_5d_codes.append("")
            ssoc_4d_codes.append("")
            continue
        
        # If SSOC is not usable, fall back to job title / skill keywords before using
        # the broad primary category.
        text_blob = " ".join([title] + skills)
        matched = None
        if isinstance(keyword_rules, list):
            for rule in keyword_rules:
                if not isinstance(rule, dict):
                    continue
                family = rule.get("family")
                keywords = rule.get("keywords", [])
                if family and isinstance(keywords, list) and _keyword_match(text_blob, keywords):
                    matched = str(family)
                    break

        if matched:
            families.append(matched)
            family_source.append("keyword")
            ssoc_5d_codes.append("")
            ssoc_4d_codes.append("")
            continue

        if isinstance(category_map, dict) and category in category_map:
            families.append(str(category_map[category]))
            family_source.append("category")
            ssoc_5d_codes.append("")
            ssoc_4d_codes.append("")
            continue

        families.append("Other")
        family_source.append("fallback")
        ssoc_5d_codes.append("")
        ssoc_4d_codes.append("")

    out = jobs.copy()
    out["role_family"] = families
    out["role_family_source"] = family_source
    out["ssoc_5d"] = ssoc_5d_codes
    out["ssoc_4d"] = ssoc_4d_codes

    ssoc_4d_official, ssoc_5d_official = _load_ssoc_title_maps()
    ssoc_5d_fallback_map = _build_ssoc_name_map(out, "ssoc_5d")
    ssoc_4d_fallback_map = _build_ssoc_name_map(out, "ssoc_4d")

    out["ssoc_5d_name"] = out["ssoc_5d"].astype(str).map(ssoc_5d_official).fillna("")
    out["ssoc_4d_name"] = out["ssoc_4d"].astype(str).map(ssoc_4d_official).fillna("")

    if ssoc_5d_fallback_map:
        miss5 = out["ssoc_5d_name"] == ""
        out.loc[miss5, "ssoc_5d_name"] = out.loc[miss5, "ssoc_5d"].map(ssoc_5d_fallback_map).fillna("")

    if ssoc_4d_fallback_map:
        miss4 = out["ssoc_4d_name"] == ""
        out.loc[miss4, "ssoc_4d_name"] = out.loc[miss4, "ssoc_4d"].map(ssoc_4d_fallback_map).fillna("")

    def _role_display_name(row: pd.Series) -> str:
        source = str(row.get("role_family_source", ""))
        if source in {"ssoc4_from_ssoc5", "ssoc2_map"}:
            name = str(row.get("ssoc_4d_name", "")).strip()
            return name if name else str(row.get("role_family", ""))
        return str(row.get("role_family", ""))

    out["ssoc_role_family"] = out["role_family"]
    out["ssoc_role_family_name"] = out.apply(_role_display_name, axis=1)
    out["ssoc_role_family_source"] = out["role_family_source"]

    role_clusters: List[str] = []
    role_cluster_sources: List[str] = []
    broad_families: List[str] = []
    cluster_rules = cluster_rules or {}

    for _, row in out.iterrows():
        # Role clusters are the curated labels used by module scoring and the UI. They
        # are designed to be easier to interpret than raw SSOC codes.
        cluster, cluster_source, broad_family = _assign_role_cluster(
            row,
            cluster_rules,
            legacy_family=str(row.get("ssoc_role_family", "")),
        )
        role_clusters.append(cluster)
        role_cluster_sources.append(cluster_source)
        broad_families.append(broad_family)

    out["role_cluster"] = role_clusters
    out["role_cluster_source"] = role_cluster_sources
    out["broad_family"] = broad_families
    out["broad_family_name"] = broad_families

    out["role_family"] = out["role_cluster"]
    out["role_family_name"] = out["role_cluster"]
    out["role_family_source"] = out["role_cluster_source"]

    diagnostics: Dict[str, float | str] = {
        "role_family_unique": float(out["role_family"].nunique()),
        "role_cluster_unique": float(out["role_cluster"].nunique()),
        "broad_family_unique": float(out["broad_family"].nunique()),
        "role_family_ssoc_rows": float(out["role_family_source"].isin(["ssoc5_exact", "ssoc4_exact"]).sum()),
        "role_family_keyword_rows": float((out["role_family_source"] == "split_rule").sum()),
        "role_family_category_rows": float((out["role_family_source"] == "legacy_family").sum()),
        "role_family_fallback_rows": float((out["role_family_source"] == "fallback").sum()),
        "jobs_with_ssoc5_rows": float((out["ssoc_5d"] != "").sum()),
        "jobs_with_ssoc4_rows": float((out["ssoc_4d"] != "").sum()),
        "jobs_unique_ssoc5": float(out.loc[out["ssoc_5d"] != "", "ssoc_5d"].nunique()),
        "jobs_unique_ssoc4": float(out.loc[out["ssoc_4d"] != "", "ssoc_4d"].nunique()),
        "jobs_unique_ssoc5_names": float(out.loc[out["ssoc_5d_name"] != "", "ssoc_5d_name"].nunique()),
        "jobs_unique_ssoc4_names": float(out.loc[out["ssoc_4d_name"] != "", "ssoc_4d_name"].nunique()),
        "ssoc4_official_map_size": float(len(ssoc_4d_official)),
        "ssoc5_official_map_size": float(len(ssoc_5d_official)),
    }

    return RoleFamilyResult(jobs=out, diagnostics=diagnostics)
