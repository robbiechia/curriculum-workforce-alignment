from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import PipelineConfig


@dataclass
class DegreeAggregationResult:
    degree_requirement_buckets: pd.DataFrame
    degree_module_map: pd.DataFrame
    degree_skill_supply: pd.DataFrame
    degree_summary: pd.DataFrame
    degree_plan_expansion_audit: pd.DataFrame
    diagnostics: Dict[str, float | str]


def _slugify(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def _degree_id(faculty_code: str, major: str) -> str:
    return f"{_slugify(faculty_code).upper()}__{_slugify(major)}"


def _normalize_module_code(code: str) -> str:
    # Runtime module scores normally use consolidated base codes for single-letter
    # suffix variants such as ACC1701A -> ACC1701.
    value = str(code or "").strip().upper().replace("\ufffd", "")
    if re.fullmatch(r"[A-Z]+\d{4}[A-Z]", value):
        return value[:-1]
    return value


def _parse_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(";") if item.strip()]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, q))


# Modules whose code matches this pattern are independent study supervision slots.
# They carry no teachable curriculum content; their scores are retrieval artifacts
# caused by generic job descriptions matching the boilerplate module text.
_INDEPENDENT_STUDY_RE = re.compile(r"^[A-Z]+4660[A-Z]*$")

# Common curriculum buckets remain visible in the degree plan and dashboard. The
# live pipeline no longer materializes the older degree-role / occupation score
# tables by default, but the helper scoring functions are retained for archival or
# ad-hoc analysis paths.
COMMON_CURRICULUM_SCORE_WEIGHT: float = 0.5


def _support_weight(evidence_count: float, prior: float) -> float:
    evidence = max(0.0, float(evidence_count))
    prior = max(0.0, float(prior))
    if evidence <= 0:
        return 0.0
    if prior <= 0:
        return 1.0
    return evidence / (evidence + prior)


def _required_bucket_columns() -> list[str]:
    return [
        "degree_id",
        "bucket_id",
        "faculty",
        "faculty_code",
        "degree",
        "major",
        "primary_major",
        "curriculum_type",
        "curriculum_credits",
        "module_type",
        "module_credits",
        "raw_modules",
        "curriculum_link",
        "curriculum_website",
        "is_unrestricted_elective",
        "is_wildcard_bucket",
        "expansion_rule",
        "eligible_module_count",
        "matched_module_count",
        "unmatched_tokens",
    ]


def _degree_module_map_columns() -> list[str]:
    return [
        "degree_id",
        "bucket_id",
        "faculty",
        "faculty_code",
        "degree",
        "major",
        "primary_major",
        "curriculum_type",
        "module_type",
        "module_credits",
        "curriculum_link",
        "curriculum_website",
        "is_unrestricted_elective",
        "expansion_rule",
        "source_token",
        "required_module_code_raw",
        "module_code",
        "module_found",
        "module_title",
        "module_profile",
        "module_faculty",
        "module_department",
        "module_credit",
        "technical_skills",
        "soft_skills",
        "top_role_cluster",
        "top_role_family_name",
        "top_role_family_name_source",
        "top_role_score",
        "top_broad_family",
    ]


def _audit_columns() -> list[str]:
    return [
        "degree_id",
        "bucket_id",
        "faculty_code",
        "degree",
        "primary_major",
        "curriculum_type",
        "module_type",
        "source_token",
        "issue_type",
        "message",
        "matched_count",
    ]


def _module_meta_frame(modules: pd.DataFrame) -> pd.DataFrame:
    module_meta = modules.copy()
    if "module_code" not in module_meta.columns:
        module_meta["module_code"] = ""
    module_meta["module_code"] = module_meta["module_code"].astype(str).str.strip().str.upper()
    for column in ["technical_skills", "soft_skills"]:
        if column in module_meta.columns:
            module_meta[column] = module_meta[column].apply(_parse_list)

    cols = [
        "module_code",
        "module_title",
        "module_profile",
        "module_faculty",
        "module_department",
        "module_credit",
        "technical_skills",
        "soft_skills",
    ]
    for col in cols:
        if col not in module_meta.columns:
            module_meta[col] = [] if col in {"technical_skills", "soft_skills"} else ""
    return module_meta[cols].drop_duplicates(subset=["module_code"]).reset_index(drop=True)


def _summary_meta_frame(module_summary: pd.DataFrame) -> pd.DataFrame:
    summary_meta = module_summary.copy()
    if "module_code" not in summary_meta.columns:
        summary_meta["module_code"] = ""
    summary_meta["module_code"] = summary_meta["module_code"].astype(str).str.strip().str.upper()

    rename_map = {
        "top_role_family": "top_role_cluster",
    }
    summary_meta = summary_meta.rename(columns={k: v for k, v in rename_map.items() if k in summary_meta.columns})
    cols = [
        "module_code",
        "top_role_cluster",
        "top_role_family_name",
        "top_role_family_name_source",
        "top_role_score",
        "top_broad_family",
    ]
    for col in cols:
        if col not in summary_meta.columns:
            summary_meta[col] = ""
    return summary_meta[cols].drop_duplicates(subset=["module_code"]).reset_index(drop=True)


def _available_module_universe(modules: pd.DataFrame, raw_modules: pd.DataFrame | None) -> tuple[set[str], set[str]]:
    raw_source = raw_modules if raw_modules is not None and not raw_modules.empty else modules
    raw_codes = {
        str(code).strip().upper().replace("\ufffd", "")
        for code in raw_source.get("module_code", pd.Series(dtype=str)).tolist()
        if str(code).strip()
    }
    consolidated_codes = {
        str(code).strip().upper().replace("\ufffd", "")
        for code in modules.get("module_code", pd.Series(dtype=str)).tolist()
        if str(code).strip()
    }
    consolidated_codes |= {_normalize_module_code(code) for code in raw_codes}
    return raw_codes, consolidated_codes


def _token_to_regex(token: str) -> tuple[re.Pattern[str] | None, str | None]:
    value = token.strip().upper().replace("\ufffd", "")
    if not value:
        return None, "empty token"
    if "%" in value:
        if value.count("%") != 1 or not value.endswith("%"):
            return None, "percent wildcard must appear once at the end"
        prefix = re.escape(value[:-1])
        return re.compile(f"^{prefix}[A-Z]+$"), None

    match = re.fullmatch(r"([A-Z]+?)([0-9X]+)([A-Z]*)", value)
    if not match:
        return None, "token is not PREFIX + digit/X code + optional suffix"
    prefix, numeric_pattern, suffix = match.groups()
    if "X" not in numeric_pattern:
        return re.compile(f"^{re.escape(value)}$"), None

    # X is a wildcard only inside the numeric code segment; suffix letters stay literal.
    digit_pattern = "".join(r"\d" if char == "X" else re.escape(char) for char in numeric_pattern)
    return re.compile(f"^{re.escape(prefix)}{digit_pattern}{re.escape(suffix)}$"), None


def _expand_token(
    token: str,
    *,
    raw_codes: set[str],
    consolidated_codes: set[str],
) -> tuple[list[tuple[str, str]], str, str | None]:
    value = token.strip().upper().replace("\ufffd", "")
    regex, error = _token_to_regex(value)
    if error:
        return [], "malformed", error

    has_wildcard = "%" in value or "X" in value
    if has_wildcard:
        raw_matches = sorted(code for code in raw_codes if regex and regex.fullmatch(code))
        normalized_matches = sorted({_normalize_module_code(code) for code in raw_matches} & consolidated_codes)
        if not normalized_matches:
            # Fall back to consolidated codes because quick/test runs may only have
            # consolidated module rows available.
            normalized_matches = sorted(code for code in consolidated_codes if regex and regex.fullmatch(code))
        return [(value, code) for code in normalized_matches], "wildcard", None

    candidates = []
    if value in consolidated_codes:
        candidates.append(value)
    normalized = _normalize_module_code(value)
    if normalized in consolidated_codes:
        candidates.append(normalized)
    candidates = sorted(dict.fromkeys(candidates))
    return [(value, code) for code in candidates], "exact", None


def _load_degree_mapping(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "degree_id",
                "faculty",
                "faculty_code",
                "major",
                "curriculum_link",
                "notes",
                "required_modules",
            ]
        )

    df = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace", dtype=str, keep_default_na=False)
    for col in ["faculty", "faculty_code", "major", "curriculum_link", "notes", "required_modules"]:
        if col not in df.columns:
            df[col] = ""
    out = df.copy()
    for col in ["faculty", "faculty_code", "major", "curriculum_link", "notes", "required_modules"]:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["degree_id"] = out.apply(
        lambda row: _degree_id(str(row.get("faculty_code", "")), str(row.get("major", ""))),
        axis=1,
    )
    return out[
        ["degree_id", "faculty", "faculty_code", "major", "curriculum_link", "notes", "required_modules"]
    ].drop_duplicates(subset=["degree_id"]).reset_index(drop=True)


def _load_degree_plan(config: PipelineConfig) -> tuple[pd.DataFrame, str]:
    plan_path = getattr(config, "degree_plan_file", Path(""))
    if plan_path and Path(plan_path).exists():
        df = pd.read_csv(plan_path, encoding="utf-8-sig", encoding_errors="replace", dtype=str, keep_default_na=False)
        df.columns = [str(col).strip() for col in df.columns]
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        return df, "degree_plan"

    mapping = _load_degree_mapping(config.degree_mapping_file)
    if mapping.empty:
        return pd.DataFrame(columns=_required_bucket_columns()), "legacy_mapping"

    rows: list[dict[str, object]] = []
    for _, row in mapping.iterrows():
        rows.append(
            {
                "faculty": row["faculty"],
                "faculty_code": row["faculty_code"],
                "degree": "",
                "primary_major": row["major"],
                "curriculum_type": "Required Modules",
                "curriculum_credits": "",
                "module_type": "Required Modules",
                "module_credits": "",
                "modules": str(row["required_modules"]),
                "curriculum_website": row["curriculum_link"],
            }
        )
    return pd.DataFrame(rows), "legacy_mapping"


def _prepare_requirement_buckets(
    plan: pd.DataFrame,
    *,
    raw_codes: set[str],
    consolidated_codes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    buckets: list[dict[str, object]] = []
    module_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for row_idx, row in plan.reset_index(drop=True).iterrows():
        faculty = str(row.get("faculty", "")).strip()
        faculty_code = str(row.get("faculty_code", "")).strip()
        degree = str(row.get("degree", "")).strip()
        major = str(row.get("primary_major", row.get("major", ""))).strip()
        curriculum_type = str(row.get("curriculum_type", "")).strip()
        module_type = str(row.get("module_type", "")).strip()
        curriculum_link = str(row.get("curriculum_website", row.get("curriculum_link", ""))).strip()
        raw_modules = str(row.get("modules", "")).strip()
        module_tokens = _parse_list(raw_modules)
        module_credits = _safe_float(row.get("module_credits", ""), 0.0)
        curriculum_credits = str(row.get("curriculum_credits", "")).strip()
        degree_id = _degree_id(faculty_code, major)
        bucket_id = f"{degree_id}__B{row_idx + 1:04d}"
        is_unrestricted = not module_tokens

        expansion_rule = "unrestricted_all" if is_unrestricted else "exact"
        expanded: list[tuple[str, str]] = []
        unmatched_tokens: list[str] = []

        if is_unrestricted:
            expanded = [("", code) for code in sorted(consolidated_codes)]
        else:
            for token in module_tokens:
                token_matches, token_rule, error = _expand_token(
                    token,
                    raw_codes=raw_codes,
                    consolidated_codes=consolidated_codes,
                )
                if token_rule == "wildcard":
                    expansion_rule = "wildcard"
                if error or not token_matches:
                    unmatched_tokens.append(token)
                    audit_rows.append(
                        {
                            "degree_id": degree_id,
                            "bucket_id": bucket_id,
                            "faculty_code": faculty_code,
                            "degree": degree,
                            "primary_major": major,
                            "curriculum_type": curriculum_type,
                            "module_type": module_type,
                            "source_token": token,
                            "issue_type": "malformed_token" if error else "unmatched_token",
                            "message": error or "No matching module code in module corpus",
                            "matched_count": 0,
                        }
                    )
                    continue
                expanded.extend(token_matches)

        deduped = sorted({(source_token, code) for source_token, code in expanded}, key=lambda item: (item[1], item[0]))
        matched_codes = sorted({code for _, code in deduped if code})
        is_wildcard = expansion_rule in {"wildcard", "unrestricted_all"}

        bucket = {
            "degree_id": degree_id,
            "bucket_id": bucket_id,
            "faculty": faculty,
            "faculty_code": faculty_code,
            "degree": degree,
            "major": major,
            "primary_major": major,
            "curriculum_type": curriculum_type,
            "curriculum_credits": curriculum_credits,
            "module_type": module_type,
            "module_credits": module_credits,
            "raw_modules": raw_modules,
            "curriculum_link": curriculum_link,
            "curriculum_website": curriculum_link,
            "is_unrestricted_elective": is_unrestricted,
            "is_wildcard_bucket": is_wildcard,
            "expansion_rule": expansion_rule,
            "eligible_module_count": len(matched_codes),
            "matched_module_count": len(matched_codes),
            "unmatched_tokens": "; ".join(unmatched_tokens),
        }
        buckets.append(bucket)

        for source_token, module_code in deduped:
            module_rows.append(
                {
                    "degree_id": degree_id,
                    "bucket_id": bucket_id,
                    "faculty": faculty,
                    "faculty_code": faculty_code,
                    "degree": degree,
                    "major": major,
                    "primary_major": major,
                    "curriculum_type": curriculum_type,
                    "module_type": module_type,
                    "module_credits": module_credits,
                    "curriculum_link": curriculum_link,
                    "curriculum_website": curriculum_link,
                    "is_unrestricted_elective": is_unrestricted,
                    "expansion_rule": expansion_rule,
                    "source_token": source_token,
                    "required_module_code_raw": source_token,
                    "module_code": module_code,
                    "module_found": bool(module_code),
                }
            )

    buckets_df = pd.DataFrame(buckets, columns=_required_bucket_columns())
    module_map = pd.DataFrame(module_rows, columns=[col for col in _degree_module_map_columns() if col in module_rows[0]] if module_rows else _degree_module_map_columns())
    audit = pd.DataFrame(audit_rows, columns=_audit_columns())
    return buckets_df, module_map, audit


def _attach_module_metadata(
    degree_module_map: pd.DataFrame,
    modules: pd.DataFrame,
    module_summary: pd.DataFrame,
) -> pd.DataFrame:
    if degree_module_map.empty:
        return pd.DataFrame(columns=_degree_module_map_columns())

    out = degree_module_map.copy()
    out["module_code"] = out["module_code"].astype(str).str.strip().str.upper()
    out = out.merge(_module_meta_frame(modules), on="module_code", how="left")
    out = out.merge(_summary_meta_frame(module_summary), on="module_code", how="left")
    out["module_found"] = out["module_title"].fillna("").astype(str).str.strip().ne("")

    for col in _degree_module_map_columns():
        if col not in out.columns:
            out[col] = ""
    return out[_degree_module_map_columns()]


def _build_plan_module_outputs(
    config: PipelineConfig,
    modules: pd.DataFrame,
    module_summary: pd.DataFrame,
    raw_modules: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    plan, source_kind = _load_degree_plan(config)
    raw_codes, consolidated_codes = _available_module_universe(modules, raw_modules)
    buckets, module_map, audit = _prepare_requirement_buckets(
        plan,
        raw_codes=raw_codes,
        consolidated_codes=consolidated_codes,
    )
    module_map = _attach_module_metadata(module_map, modules, module_summary)

    if not module_map.empty:
        matched_counts = (
            module_map[module_map["module_found"]]
            .groupby("bucket_id")["module_code"]
            .nunique()
            .rename("matched_module_count_actual")
        )
        buckets = buckets.merge(matched_counts, on="bucket_id", how="left")
        buckets["matched_module_count"] = buckets["matched_module_count_actual"].fillna(0).astype(int)
        buckets = buckets.drop(columns=["matched_module_count_actual"])
    return buckets, module_map, audit, source_kind


def _numeric_module_credit(value: object) -> float:
    credit = _safe_float(value, 0.0)
    return credit if credit > 0 else 4.0


def _score_requirement_bucket(group: pd.DataFrame, score_col: str, mode: str, required_credits: float) -> tuple[float, list[str], int]:
    if group.empty:
        return 0.0, [], 0
    unique_modules = (
        group.sort_values(score_col, ascending=False)
        .drop_duplicates(subset=["module_code"])
        .copy()
    )
    unique_modules["module_credit_num"] = unique_modules["module_credit"].apply(_numeric_module_credit)

    if mode == "expected":
        values = unique_modules[score_col].to_numpy(dtype=float)
        return float(values.mean()) if values.size else 0.0, unique_modules["module_code"].astype(str).tolist(), int(len(unique_modules))

    selected = []
    selected_credits = 0.0
    credit_target = required_credits if required_credits > 0 else 4.0
    for _, row in unique_modules.iterrows():
        selected.append(row)
        selected_credits += _numeric_module_credit(row.get("module_credit"))
        if selected_credits >= credit_target:
            break
    if not selected:
        return 0.0, [], 0
    selected_df = pd.DataFrame(selected)
    weights = selected_df["module_credit_num"].to_numpy(dtype=float)
    values = selected_df[score_col].to_numpy(dtype=float)
    return float(np.average(values, weights=weights)) if weights.sum() else float(values.mean()), selected_df["module_code"].astype(str).tolist(), int(len(selected_df))


def _elective_potential(
    source_df: pd.DataFrame,
    *,
    group_cols: list[str],
    credit_budget: float,
) -> dict[tuple[str, ...], tuple[float, str]]:
    if credit_budget <= 0 or source_df.empty:
        return {}
    source = source_df.copy()
    source["module_credit_num"] = source.get("module_credit", pd.Series(4.0, index=source.index)).apply(_numeric_module_credit)
    out: dict[tuple[str, ...], tuple[float, str]] = {}
    for keys, group in source.groupby(group_cols, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        key_tuple = tuple(str(value) for value in key_tuple)
        selected = []
        selected_credits = 0.0
        for _, row in group.sort_values("role_score", ascending=False).drop_duplicates("module_code").iterrows():
            selected.append(row)
            selected_credits += _numeric_module_credit(row.get("module_credit"))
            if selected_credits >= credit_budget:
                break
        if not selected:
            continue
        selected_df = pd.DataFrame(selected)
        weights = selected_df["module_credit_num"].to_numpy(dtype=float)
        values = selected_df["role_score"].to_numpy(dtype=float)
        score = float(np.average(values, weights=weights)) if weights.sum() else float(values.mean())
        out[key_tuple] = (score, "; ".join(selected_df["module_code"].astype(str).head(10).tolist()))
    return out


def _compute_effective_prior(
    degree_id: str,
    degree_module_map: pd.DataFrame,
    total_required_bucket_counts: Dict[str, int],
) -> float:
    # Derive the support prior from the degree's own curriculum coverage rather than a
    # fixed constant. Degrees with poor corpus coverage (many required modules missing
    # from NUSMods) receive a larger prior, which reduces their support weight. Degrees
    # with full coverage and fully-scored buckets approach support_weight = 1.0.
    non_unrestricted = degree_module_map[
        (degree_module_map["degree_id"] == degree_id)
        & ~degree_module_map["is_unrestricted_elective"].astype(bool)
    ]
    total_modules = non_unrestricted["module_code"].nunique()
    matched_modules = non_unrestricted[non_unrestricted["module_found"].astype(bool)]["module_code"].nunique()
    matched_share = matched_modules / total_modules if total_modules > 0 else 0.0
    total_buckets = float(total_required_bucket_counts.get(degree_id, 1))
    return (1.0 - matched_share) * total_buckets


def _aggregate_degree_scores(
    degree_requirement_buckets: pd.DataFrame,
    degree_module_map: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    group_cols: List[str],
    rank_col_name: str,
    score_col_name: str,
    constraints: ModuleConstraints | None = None,
) -> pd.DataFrame:
    base_cols = [
        "degree_id",
        "faculty",
        "faculty_code",
        "degree",
        "major",
        "primary_major",
        "score_mode",
        *group_cols,
        score_col_name,
        "raw_degree_score",
        "support_weight",
        "evidence_bucket_count",
        "evidence_module_count",
        "required_credit_count",
        "top_contributing_modules",
        "elective_half_potential_score",
        "elective_full_potential_score",
        "score_with_half_electives",
        "score_with_full_electives",
        "elective_half_modules",
        "elective_full_modules",
        rank_col_name,
    ]
    if degree_requirement_buckets.empty or degree_module_map.empty or source_df.empty:
        return pd.DataFrame(columns=base_cols)

    source = source_df.copy()
    if "role_score" not in source.columns:
        return pd.DataFrame(columns=base_cols)
    source["module_code"] = source["module_code"].astype(str).str.strip().str.upper()
    module_credits = degree_module_map[["module_code", "module_credit"]].drop_duplicates("module_code")
    source = source.merge(module_credits, on="module_code", how="left")

    # Pre-compute total required (non-unrestricted) bucket count per degree for the
    # data-adaptive prior calculation.
    non_unrestricted_buckets = degree_requirement_buckets[
        ~degree_requirement_buckets["is_unrestricted_elective"].astype(bool)
    ]
    total_required_bucket_counts: Dict[str, int] = (
        non_unrestricted_buckets.groupby("degree_id")["bucket_id"]
        .nunique()
        .to_dict()
    )

    matched = degree_module_map[
        degree_module_map["module_found"] & ~degree_module_map["is_unrestricted_elective"].astype(bool)
    ].copy()
    if matched.empty:
        return pd.DataFrame(columns=base_cols)

    joined = matched.merge(source, on="module_code", how="inner", suffixes=("", "_score"))
    if joined.empty:
        return pd.DataFrame(columns=base_cols)

    # Exclude independent study supervision slots (e.g. GE4660, HY4660HM).
    # These modules have boilerplate descriptions and produce spurious role scores.
    joined = joined[~joined["module_code"].astype(str).str.match(_INDEPENDENT_STUDY_RE)].copy()
    if joined.empty:
        return pd.DataFrame(columns=base_cols)

    bucket_meta = degree_requirement_buckets.set_index("bucket_id").to_dict("index")
    unrestricted_credits = (
        degree_requirement_buckets[degree_requirement_buckets["is_unrestricted_elective"].astype(bool)]
        .groupby("degree_id")["module_credits"]
        .sum()
        .to_dict()
    )
    all_degree_meta = (
        degree_requirement_buckets.groupby("degree_id", as_index=False)
        .first()[["degree_id", "faculty", "faculty_code", "degree", "major", "primary_major"]]
        .set_index("degree_id")
        .to_dict("index")
    )

    all_targets = source[group_cols + ["module_code", "role_score", "module_credit"]].copy()

    # Pre-compute constraint sets once per degree_id (not per role/ssoc group × mode).
    # reachable_set  = all modules prereq-reachable from the degree's required curriculum
    # precluded_set  = all modules precluded by any module in that curriculum
    # Both sets are used to gate the elective candidate pool in the inner loop via
    # cheap set-membership checks — no repeated graph traversal per group.
    _constraint_cache: dict[str, tuple[set[str], set[str]]] = {}
    if constraints is not None:
        degree_req_modules = (
            degree_module_map[
                degree_module_map["module_found"].astype(bool)
                & ~degree_module_map["is_unrestricted_elective"].astype(bool)
            ]
            .groupby("degree_id")["module_code"]
            .apply(lambda s: set(s.str.strip().str.upper()))
            .to_dict()
        )
        for did, mods in degree_req_modules.items():
            _constraint_cache[str(did)] = (
                constraints.reachable_from(mods),
                constraints.precluded_by_curriculum(mods),
            )

    rows: list[dict[str, object]] = []
    degree_group_cols = ["degree_id", *group_cols]
    for mode in ["expected", "best_case"]:
        for keys, group in joined.groupby(degree_group_cols, dropna=False):
            degree_id, *group_values = keys if isinstance(keys, tuple) else (keys,)
            bucket_scores = []
            module_score_map: dict[str, float] = {}  # module_code -> best role_score seen
            evidence_modules: set[str] = set()

            for bucket_id, bucket_group in group.groupby("bucket_id"):
                bucket = bucket_meta.get(bucket_id, {})
                required_credits = _safe_float(bucket.get("module_credits"), 0.0)
                bucket_score, modules_used, module_count = _score_requirement_bucket(
                    bucket_group,
                    "role_score",
                    mode,
                    required_credits,
                )
                if module_count <= 0:
                    continue
                # Fall back to 4 MC when module_credits is missing from the degree plan
                # rather than picking a credit value from an arbitrary row in the group.
                weight = required_credits if required_credits > 0 else 4.0
                # Common curriculum buckets are generic by design and inflate the
                # Education/Training role score. Downweight their credit contribution.
                curriculum_type = str(bucket.get("curriculum_type", "")).strip()
                if "Common Curriculum" in curriculum_type:
                    weight *= COMMON_CURRICULUM_SCORE_WEIGHT
                bucket_scores.append((bucket_score, weight))
                evidence_modules.update(modules_used)
                # Track best role_score per contributing module so the final list can be
                # sorted by actual contribution rather than bucket iteration order.
                score_lookup = (
                    bucket_group.drop_duplicates("module_code")
                    .set_index("module_code")["role_score"]
                    .to_dict()
                )
                for mc in modules_used:
                    mc_score = float(score_lookup.get(mc, 0.0))
                    if mc not in module_score_map or mc_score > module_score_map[mc]:
                        module_score_map[mc] = mc_score

            if not bucket_scores:
                continue

            weights = np.array([weight for _, weight in bucket_scores], dtype=float)
            values = np.array([score for score, _ in bucket_scores], dtype=float)
            raw_score = float(np.average(values, weights=weights)) if weights.sum() else float(values.mean())
            effective_prior = _compute_effective_prior(degree_id, degree_module_map, total_required_bucket_counts)
            support_weight = _support_weight(len(bucket_scores), effective_prior)
            degree_score = raw_score * support_weight
            required_credits = float(weights.sum())
            elective_credit_total = float(unrestricted_credits.get(degree_id, 0.0) or 0.0)

            row = {
                "degree_id": degree_id,
                **all_degree_meta.get(degree_id, {}),
                "score_mode": mode,
                score_col_name: degree_score,
                "raw_degree_score": raw_score,
                "support_weight": support_weight,
                "evidence_bucket_count": float(len(bucket_scores)),
                "evidence_module_count": float(len(evidence_modules)),
                "required_credit_count": required_credits,
                "top_contributing_modules": "; ".join(
                    mc for mc, _ in sorted(module_score_map.items(), key=lambda x: x[1], reverse=True)
                ),
                "elective_half_potential_score": 0.0,
                "elective_full_potential_score": 0.0,
                "score_with_half_electives": degree_score,
                "score_with_full_electives": degree_score,
                "elective_half_modules": "",
                "elective_full_modules": "",
            }
            for col, value in zip(group_cols, group_values):
                row[col] = value

            if elective_credit_total > 0:
                target_filter = pd.Series(True, index=all_targets.index)
                for col, value in zip(group_cols, group_values):
                    target_filter &= all_targets[col].astype(str).eq(str(value))
                target_source = all_targets[target_filter & ~all_targets["module_code"].isin(evidence_modules)].copy()

                # Gate elective candidates using the pre-computed constraint cache.
                # O(candidates) set-membership check — no graph traversal here.
                if _constraint_cache and str(degree_id) in _constraint_cache:
                    reachable_set, precluded_set = _constraint_cache[str(degree_id)]
                    candidate_codes = target_source["module_code"].str.strip().str.upper()
                    allowed = (set(candidate_codes) & reachable_set) - precluded_set
                    target_source = target_source[candidate_codes.isin(allowed)].copy()
                half = _elective_potential(target_source, group_cols=group_cols, credit_budget=elective_credit_total / 2.0)
                full = _elective_potential(target_source, group_cols=group_cols, credit_budget=elective_credit_total)
                target_key = tuple(str(v) for v in group_values)
                half_score, half_modules = half.get(target_key, (0.0, ""))
                full_score, full_modules = full.get(target_key, (0.0, ""))
                row["elective_half_potential_score"] = half_score
                row["elective_full_potential_score"] = full_score
                # Combine raw scores before applying support weight so elective modules
                # are not mixed with an already-shrunk base score.
                half_combined_credits = required_credits + (elective_credit_total / 2.0)
                full_combined_credits = required_credits + elective_credit_total
                row["score_with_half_electives"] = (
                    (
                        ((raw_score * required_credits) + (half_score * (elective_credit_total / 2.0)))
                        / half_combined_credits
                    ) * support_weight
                    if half_combined_credits > 0
                    else degree_score
                )
                row["score_with_full_electives"] = (
                    (
                        ((raw_score * required_credits) + (full_score * elective_credit_total))
                        / full_combined_credits
                    ) * support_weight
                    if full_combined_credits > 0
                    else degree_score
                )
                row["elective_half_modules"] = half_modules
                row["elective_full_modules"] = full_modules

            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=base_cols)
    out[rank_col_name] = out.groupby(["degree_id", "score_mode"])[score_col_name].rank(
        ascending=False,
        method="dense",
    )
    sort_cols = ["degree_id", "score_mode", score_col_name]
    ascending = [True, True, False]
    return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _build_degree_skill_supply(degree_requirement_buckets: pd.DataFrame, degree_module_map: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "degree_id",
        "faculty",
        "faculty_code",
        "degree",
        "major",
        "primary_major",
        "skill",
        "module_count",
        "matched_required_module_count",
        "supply_score",
        "module_codes",
    ]
    if degree_requirement_buckets.empty or degree_module_map.empty:
        return pd.DataFrame(columns=cols)

    matched = degree_module_map[
        degree_module_map["module_found"] & ~degree_module_map["is_unrestricted_elective"].astype(bool)
    ].copy()
    if matched.empty:
        return pd.DataFrame(columns=cols)
    matched["technical_skills"] = matched["technical_skills"].apply(_parse_list)

    bucket_counts = matched.groupby("bucket_id")["module_code"].nunique().rename("bucket_module_count")
    matched = matched.merge(bucket_counts, on="bucket_id", how="left")
    matched["bucket_module_count"] = matched["bucket_module_count"].replace(0, np.nan)
    matched["module_credits_num"] = matched["module_credits"].apply(_safe_float)
    matched["skill_weight"] = matched["module_credits_num"] / matched["bucket_module_count"].fillna(1)

    total_required_credits = (
        degree_requirement_buckets[~degree_requirement_buckets["is_unrestricted_elective"].astype(bool)]
        .groupby("degree_id")["module_credits"]
        .sum()
        .rename("required_credit_count")
    )
    matched_counts = matched.groupby("degree_id")["module_code"].nunique().rename("matched_required_module_count")

    rows: list[dict[str, object]] = []
    for _, row in matched.iterrows():
        for skill in sorted(set(_parse_list(row.get("technical_skills", [])))):
            rows.append(
                {
                    "degree_id": row["degree_id"],
                    "faculty": row["faculty"],
                    "faculty_code": row["faculty_code"],
                    "degree": row["degree"],
                    "major": row["major"],
                    "primary_major": row["primary_major"],
                    "module_code": row["module_code"],
                    "skill": skill,
                    "skill_weight": float(row["skill_weight"] or 0.0),
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)

    supply = pd.DataFrame(rows)
    grouped = (
        supply.groupby(["degree_id", "faculty", "faculty_code", "degree", "major", "primary_major", "skill"], as_index=False)
        .agg(
            module_count=("module_code", "nunique"),
            weighted_module_credits=("skill_weight", "sum"),
            module_codes=("module_code", lambda s: "; ".join(sorted(set(map(str, s))))),
        )
    )
    grouped = grouped.merge(total_required_credits, on="degree_id", how="left")
    grouped = grouped.merge(matched_counts, on="degree_id", how="left")
    grouped["required_credit_count"] = grouped["required_credit_count"].replace(0, np.nan)
    grouped["matched_required_module_count"] = grouped["matched_required_module_count"].fillna(0).astype(int)
    grouped["supply_score"] = (grouped["weighted_module_credits"] / grouped["required_credit_count"]).fillna(0.0)
    return grouped[cols].sort_values(
        ["degree_id", "supply_score", "module_count", "skill"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def _build_degree_skill_gaps(
    jobs: pd.DataFrame,
    degree_scores: pd.DataFrame,
    degree_skill_supply: pd.DataFrame,
    *,
    entity_cols: List[str],
    score_col: str,
    rank_col: str,
    demand_skill_top_n: int,
) -> pd.DataFrame:
    base_cols = [
        "degree_id",
        "faculty",
        "faculty_code",
        "degree",
        "major",
        "primary_major",
        *entity_cols,
        score_col,
        rank_col,
        "skill",
        "demand_weight",
        "is_covered",
        "covering_module_count",
        "module_codes",
    ]
    if jobs.empty or degree_scores.empty or degree_skill_supply.empty:
        return pd.DataFrame(columns=base_cols)
    degree_scores = degree_scores[degree_scores.get("score_mode", "expected").astype(str).eq("expected")].copy()
    if degree_scores.empty:
        return pd.DataFrame(columns=base_cols)
    if not set(entity_cols).issubset(jobs.columns) or not set(entity_cols).issubset(degree_scores.columns):
        return pd.DataFrame(columns=base_cols)

    demand_by_entity_skill: Dict[tuple[str, ...], Counter] = defaultdict(Counter)
    entity_display_map: Dict[tuple[str, ...], Dict[str, str]] = {}
    for _, row in jobs.iterrows():
        entity_key = tuple(str(row.get(col, "")).strip() for col in entity_cols)
        if not entity_key or not entity_key[0]:
            continue
        entity_display_map[entity_key] = {col: str(row.get(col, "")).strip() for col in entity_cols}
        for skill in set(_parse_list(row.get("technical_skills", []))):
            demand_by_entity_skill[entity_key][skill] += 1.0

    supply_lookup = degree_skill_supply.set_index(["degree_id", "skill"]).to_dict("index")
    rows: List[Dict[str, object]] = []
    for _, row in degree_scores.iterrows():
        entity_key = tuple(str(row.get(col, "")).strip() for col in entity_cols)
        demand_counter = demand_by_entity_skill.get(entity_key, Counter())
        if not demand_counter:
            continue
        demand_total = sum(demand_counter.values()) or 1.0
        top_skills = demand_counter.most_common(max(1, int(demand_skill_top_n)))
        entity_values = entity_display_map.get(entity_key, {col: value for col, value in zip(entity_cols, entity_key)})
        for skill, demand_value in top_skills:
            # demand_weight: how frequently this skill appears across job postings for
            # this role (fraction of total skill mentions in the role's job postings).
            demand_weight = float(demand_value / demand_total)
            supply_meta = supply_lookup.get((str(row["degree_id"]), skill), {})
            covering_module_count = int(supply_meta.get("module_count", 0) or 0)
            rows.append(
                {
                    "degree_id": row["degree_id"],
                    "faculty": row["faculty"],
                    "faculty_code": row["faculty_code"],
                    "degree": row.get("degree", ""),
                    "major": row["major"],
                    "primary_major": row.get("primary_major", row["major"]),
                    **entity_values,
                    score_col: float(row.get(score_col, 0.0)),
                    rank_col: float(row.get(rank_col, np.nan)),
                    "skill": skill,
                    "demand_weight": demand_weight,
                    "is_covered": covering_module_count > 0,
                    "covering_module_count": covering_module_count,
                    "module_codes": str(supply_meta.get("module_codes", "") or ""),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=base_cols)
    # Sort: high-demand uncovered skills first (is_covered asc puts False before True),
    # then by demand_weight descending within each degree-role pair.
    return out.sort_values(
        ["degree_id", rank_col, "is_covered", "demand_weight"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def _build_degree_role_skill_gaps(
    jobs: pd.DataFrame,
    degree_role_scores: pd.DataFrame,
    degree_skill_supply: pd.DataFrame,
    demand_skill_top_n: int,
) -> pd.DataFrame:
    return _build_degree_skill_gaps(
        jobs=jobs,
        degree_scores=degree_role_scores,
        degree_skill_supply=degree_skill_supply,
        entity_cols=["role_family", "role_family_name"],
        score_col="degree_role_score",
        rank_col="role_rank_within_degree",
        demand_skill_top_n=demand_skill_top_n,
    )


def _build_degree_ssoc5_skill_gaps(
    jobs: pd.DataFrame,
    degree_ssoc5_scores: pd.DataFrame,
    degree_skill_supply: pd.DataFrame,
    demand_skill_top_n: int,
) -> pd.DataFrame:
    return _build_degree_skill_gaps(
        jobs=jobs,
        degree_scores=degree_ssoc5_scores,
        degree_skill_supply=degree_skill_supply,
        entity_cols=["ssoc_4d", "ssoc_4d_name", "ssoc_5d", "ssoc_5d_name"],
        score_col="degree_ssoc5_score",
        rank_col="ssoc5_rank_within_degree",
        demand_skill_top_n=demand_skill_top_n,
    )


def _build_degree_summary(
    degree_requirement_buckets: pd.DataFrame,
    degree_module_map: pd.DataFrame,
    degree_role_scores: pd.DataFrame,
    degree_ssoc5_scores: pd.DataFrame,
) -> pd.DataFrame:
    summary_cols = [
        "degree_id",
        "faculty",
        "faculty_code",
        "degree",
        "major",
        "primary_major",
        "curriculum_link",
        "curriculum_website",
        "required_bucket_count",
        "required_module_count",
        "matched_required_module_count",
        "missing_required_module_count",
        "matched_required_module_share",
        "top_role_cluster",
        "top_role_cluster_name",
        "top_role_cluster_score",
        "top_role_cluster_best_case",
        "top_role_cluster_best_case_score",
        "top_ssoc5",
        "top_ssoc5_name",
        "top_ssoc5_score",
        "top_ssoc5_best_case",
        "top_ssoc5_best_case_name",
        "top_ssoc5_best_case_score",
    ]
    if degree_requirement_buckets.empty:
        return pd.DataFrame(columns=summary_cols)

    summary = (
        degree_requirement_buckets.groupby("degree_id", as_index=False)
        .agg(
            faculty=("faculty", "first"),
            faculty_code=("faculty_code", "first"),
            degree=("degree", "first"),
            major=("major", "first"),
            primary_major=("primary_major", "first"),
            curriculum_link=("curriculum_link", "first"),
            curriculum_website=("curriculum_website", "first"),
            required_bucket_count=("bucket_id", "nunique"),
        )
    )

    non_unrestricted = degree_module_map[~degree_module_map["is_unrestricted_elective"].astype(bool)].copy()
    if not non_unrestricted.empty:
        unique_required_modules = (
            non_unrestricted.groupby(["degree_id", "module_code"], as_index=False)
            .agg(module_found=("module_found", "max"))
        )
        module_counts = unique_required_modules.groupby("degree_id", as_index=False).agg(
            required_module_count=("module_code", "nunique"),
            matched_required_module_count=("module_found", lambda s: int(sum(bool(v) for v in s))),
        )
        module_counts["missing_required_module_count"] = (
            module_counts["required_module_count"] - module_counts["matched_required_module_count"]
        )
        module_counts["matched_required_module_share"] = np.where(
            module_counts["required_module_count"] > 0,
            module_counts["matched_required_module_count"] / module_counts["required_module_count"],
            0.0,
        )
        summary = summary.merge(module_counts, on="degree_id", how="left")

    if not degree_role_scores.empty:
        expected = degree_role_scores[degree_role_scores["score_mode"].eq("expected")]
        if not expected.empty:
            top_role = (
                expected.sort_values(["degree_role_score", "role_family"], ascending=[False, True])
                .groupby("degree_id", as_index=False)
                .head(1)[["degree_id", "role_family", "role_family_name", "degree_role_score"]]
                .rename(
                    columns={
                        "role_family": "top_role_cluster",
                        "role_family_name": "top_role_cluster_name",
                        "degree_role_score": "top_role_cluster_score",
                    }
                )
            )
            summary = summary.merge(top_role, on="degree_id", how="left")
        best_case = degree_role_scores[degree_role_scores["score_mode"].eq("best_case")]
        if not best_case.empty:
            top_role_best = (
                best_case.sort_values(["degree_role_score", "role_family"], ascending=[False, True])
                .groupby("degree_id", as_index=False)
                .head(1)[["degree_id", "role_family", "degree_role_score"]]
                .rename(
                    columns={
                        "role_family": "top_role_cluster_best_case",
                        "degree_role_score": "top_role_cluster_best_case_score",
                    }
                )
            )
            summary = summary.merge(top_role_best, on="degree_id", how="left")

    if not degree_ssoc5_scores.empty:
        expected = degree_ssoc5_scores[degree_ssoc5_scores["score_mode"].eq("expected")]
        if not expected.empty:
            top_ssoc5 = (
                expected.sort_values(["degree_ssoc5_score", "ssoc_5d_name"], ascending=[False, True])
                .groupby("degree_id", as_index=False)
                .head(1)[["degree_id", "ssoc_5d", "ssoc_5d_name", "degree_ssoc5_score"]]
                .rename(
                    columns={
                        "ssoc_5d": "top_ssoc5",
                        "ssoc_5d_name": "top_ssoc5_name",
                        "degree_ssoc5_score": "top_ssoc5_score",
                    }
                )
            )
            summary = summary.merge(top_ssoc5, on="degree_id", how="left")
        best_case = degree_ssoc5_scores[degree_ssoc5_scores["score_mode"].eq("best_case")]
        if not best_case.empty:
            top_ssoc5_best = (
                best_case.sort_values(["degree_ssoc5_score", "ssoc_5d_name"], ascending=[False, True])
                .groupby("degree_id", as_index=False)
                .head(1)[["degree_id", "ssoc_5d", "ssoc_5d_name", "degree_ssoc5_score"]]
                .rename(
                    columns={
                        "ssoc_5d": "top_ssoc5_best_case",
                        "ssoc_5d_name": "top_ssoc5_best_case_name",
                        "degree_ssoc5_score": "top_ssoc5_best_case_score",
                    }
                )
            )
            summary = summary.merge(top_ssoc5_best, on="degree_id", how="left")

    for col in summary_cols:
        if col not in summary.columns:
            summary[col] = 0 if col.endswith("_count") or col.endswith("_share") else ""
    return summary[summary_cols]


def build_degree_outputs(
    config: PipelineConfig,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    module_summary: pd.DataFrame,
    module_role_scores: pd.DataFrame,
    module_ssoc5_scores: pd.DataFrame,
    raw_modules: pd.DataFrame | None = None,
) -> DegreeAggregationResult:
    # Degree outputs are a pure aggregation layer over existing module outputs.
    degree_requirement_buckets, degree_module_map, audit, source_kind = _build_plan_module_outputs(
        config=config,
        modules=modules,
        module_summary=module_summary,
        raw_modules=raw_modules,
    )
    degree_skill_supply = _build_degree_skill_supply(degree_requirement_buckets, degree_module_map)

    # The current dashboard derives role analysis directly from module-level scores and
    # curriculum buckets. We therefore keep the structural degree outputs, but stop
    # materializing the older degree-role / SSOC score tables and their downstream
    # gap tables in the default live pipeline.
    degree_summary = _build_degree_summary(
        degree_requirement_buckets=degree_requirement_buckets,
        degree_module_map=degree_module_map,
        degree_role_scores=pd.DataFrame(),
        degree_ssoc5_scores=pd.DataFrame(),
    )

    diagnostics: Dict[str, float | str] = {
        "degree_source_kind": source_kind,
        "degree_plan_file": str(getattr(config, "degree_plan_file", "")),
        "degree_mapping_file": str(config.degree_mapping_file),
        "degrees_rows": float(degree_requirement_buckets["degree_id"].nunique()) if not degree_requirement_buckets.empty else 0.0,
        "degree_requirement_bucket_rows": float(len(degree_requirement_buckets)),
        "degree_module_map_rows": float(len(degree_module_map)),
        "degree_role_rows": 0.0,
        "degree_ssoc5_rows": 0.0,
        "degree_skill_supply_rows": float(len(degree_skill_supply)),
        "degree_role_skill_gap_rows": 0.0,
        "degree_ssoc5_skill_gap_rows": 0.0,
        "degree_summary_rows": float(len(degree_summary)),
        "degree_plan_expansion_audit_rows": float(len(audit)),
    }
    if not degree_module_map.empty:
        matched = degree_module_map["module_found"].sum()
        diagnostics["degree_required_module_match_share"] = float(matched / len(degree_module_map))
        diagnostics["degree_unrestricted_module_map_rows"] = float(
            degree_module_map["is_unrestricted_elective"].astype(bool).sum()
        )

    return DegreeAggregationResult(
        degree_requirement_buckets=degree_requirement_buckets,
        degree_module_map=degree_module_map,
        degree_skill_supply=degree_skill_supply,
        degree_summary=degree_summary,
        degree_plan_expansion_audit=audit,
        diagnostics=diagnostics,
    )
