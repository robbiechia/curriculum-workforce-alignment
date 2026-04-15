from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

from ..config.settings import PipelineConfig, read_yaml_json
from data_utils.db_utils import read_table



NOISY_SKILL_TOKENS = {
    "a",
    "an",
    "and",
    "or",
    "s",
    "the",
}
VALID_SINGLE_CHAR_SKILLS = {"r"}
MIN_JOB_SKILL_SUPPORT_FOR_MODULES = 2
SOFT_SKILL_MIN_SCORE = 0.25
SOFT_SKILL_MAX_ITEMS = 5
SOFT_SKILL_MARKERS: Dict[str, Tuple[str, ...]] = {
    "communication skills": (
        "communication",
        "presentation",
        "writing",
        "report",
    ),
    "team player": (
        "team",
        "teamwork",
        "collaboration",
        "collaborative",
    ),
    "leadership": (
        "leadership",
        "mentoring",
        "mentor",
    ),
    "interpersonal skills": (
        "interpersonal",
        "stakeholder",
        "client",
        "negotiation",
        "networking",
        "empathy",
        "customer service",
        "relationship",
    ),
    "analytical skills": (
        "analysis",
        "analytical",
        "critical thinking",
        "reasoning",
        "research",
    ),
    "problem solving": (
        "problem solving",
        "problem-solving",
        "troubleshooting",
        "debugging",
    ),
    "management skills": (
        "management",
        "planning",
        "coordination",
        "organizing",
        "project management",
    ),
    "able to work independently": (
        "independent",
        "initiative",
        "adaptability",
        "flexibility",
        "resilience",
        "time management",
        "autonomous",
        "self motivated",
        "self-motivated",
    ),
}
WORKLOAD_SOFT_SKILL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "communication skills": {
        "tutorial": 0.4,
        "project": 0.4,
        "lecture": 0.1,
        "preparation": 0.1,
    },
    "team player": {
        "project": 0.6,
        "tutorial": 0.2,
        "lab": 0.2,
    },
    "leadership": {
        "project": 0.35,
        "tutorial": 0.15,
    },
    "interpersonal skills": {
        "tutorial": 0.4,
        "project": 0.4,
        "lab": 0.2,
    },
    "analytical skills": {
        "lab": 0.4,
        "tutorial": 0.3,
        "preparation": 0.2,
        "lecture": 0.1,
    },
    "problem solving": {
        "lab": 0.5,
        "project": 0.3,
        "tutorial": 0.2,
    },
    "management skills": {
        "project": 0.5,
        "preparation": 0.3,
        "tutorial": 0.2,
    },
    "able to work independently": {
        "preparation": 0.7,
        "lecture": 0.2,
        "project": 0.1,
    },
}


@dataclass
class SkillTaxonomyResult:
    jobs: pd.DataFrame
    modules: pd.DataFrame
    skill_channel_map: Dict[str, str]
    diagnostics: Dict[str, float | str]



def _clean_token(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"[^a-z0-9\s\+\#]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t



def _normalize_skill(skill: str, alias_map: Dict[str, str]) -> str:
    token = _clean_token(skill)
    if not token:
        return ""
    return alias_map.get(token, token)


def _is_valid_skill(skill: str) -> bool:
    token = _clean_token(skill)
    if not token:
        return False
    if token in NOISY_SKILL_TOKENS:
        return False
    if len(token) == 1 and token not in VALID_SINGLE_CHAR_SKILLS:
        return False
    return True



def load_skill_aliases(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    payload = read_yaml_json(path)
    if not isinstance(payload, dict):
        return {}
    aliases: Dict[str, str] = {}
    for k, v in payload.items():
        key = _clean_token(str(k))
        value = _clean_token(str(v))
        if not key or not value:
            continue
        if not _is_valid_skill(value):
            continue
        aliases[key] = value
    return aliases



def load_skillsfuture_mapping() -> pd.DataFrame:
    df = read_table("skillsfuture_mapping")
    df = df.rename(columns={c: c for c in ["skill_norm", "channel", "framework_cluster", "skillsfuture_note"]})

    df["skill_norm"] = df["skill_norm"].apply(lambda v: _clean_token(str(v or "")))
    df["channel"] = df["channel"].apply(lambda v: _clean_token(str(v or "")))
    df["framework_cluster"] = df["framework_cluster"].apply(lambda v: str(v or "").strip())
    df["skillsfuture_note"] = df["skillsfuture_note"].apply(lambda v: str(v or "").strip())

    df = df[df["skill_norm"].notna() & (df["skill_norm"] != "")]
    df = df[df["skill_norm"].apply(_is_valid_skill)]
    if df.empty:
        return pd.DataFrame(columns=["skill_norm", "channel", "framework_cluster", "skillsfuture_note"])
    return df.drop_duplicates(subset=["skill_norm"]).reset_index(drop=True)



def _build_channel_map(skillsfuture_df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for _, row in skillsfuture_df.iterrows():
        skill = _clean_token(str(row.get("skill_norm", "")))
        channel = _clean_token(str(row.get("channel", "")))
        if not skill:
            continue
        if channel not in {"technical", "transferable"}:
            channel = "technical"
        out[skill] = channel
    return out



def _heuristic_channel(skill: str) -> str:
    transferable_markers = [
        "communication",
        "team",
        "stakeholder",
        "leadership",
        "problem solving",
        "critical thinking",
        "presentation",
        "report",
        "adaptability",
        "time management",
        "customer service",
        "empathy",
        "resilience",
        "emotional intelligence",
        "networking",
        "conflict resolution",
        "creativity",
        "flexibility",
        "initiative",
        "mentoring",
        "cultural awareness",
        "collaboration",
        "interpersonal",
        "negotiation",
        "writing",
        "analysis",
        "client",
        "project",
    ]
    for marker in transferable_markers:
        if marker in skill:
            return "transferable"
    return "technical"



def _split_channels(
    skills: Iterable[str], alias_map: Dict[str, str], channel_map: Dict[str, str]
) -> Tuple[List[str], List[str], Set[str]]:
    tech: List[str] = []
    transfer: List[str] = []
    unmapped: Set[str] = set()

    for raw in skills:
        s = _normalize_skill(str(raw), alias_map)
        if not s or not _is_valid_skill(s):
            continue
        if s not in channel_map:
            unmapped.add(s)
        channel = channel_map.get(s, _heuristic_channel(s))
        if channel == "transferable":
            transfer.append(s)
        else:
            tech.append(s)

    return sorted(set(tech)), sorted(set(transfer)), unmapped



def _extract_mentions(text: str, known_skills: Set[str]) -> List[str]:
    # Module skills are inferred from explicit mentions in text, not from a separate
    # curated module-skills table.
    cleaned = f" {_clean_token(text)} "
    out = []
    for skill in known_skills:
        if not skill or not _is_valid_skill(skill):
            continue
        needle = f" {skill} "
        if needle in cleaned:
            out.append(skill)
    return sorted(set(out))



def _extract_transferable_cues(text: str, cues: List[str]) -> List[str]:
    cleaned = _clean_token(text)
    out = []
    for cue in cues:
        cue_clean = _clean_token(cue)
        if not cue_clean:
            continue
        if cue_clean in cleaned:
            out.append(cue_clean)
    return sorted(set(out))



def _count_nonempty_lists(values: pd.Series) -> float:
    return float(sum(1 for value in values if isinstance(value, list) and len(value) > 0))


def _build_known_skills(
    channel_map: Dict[str, str],
    jobs_tech: List[List[str]],
    jobs_trans: List[List[str]],
) -> Set[str]:
    # Limit module extraction to a controlled vocabulary sourced from the taxonomy and
    # recurring job skills, so the module side does not invent arbitrary phrases.
    known_skills: Set[str] = {skill for skill in channel_map.keys() if _is_valid_skill(skill)}
    job_skill_counts: Counter[str] = Counter()
    for skills in jobs_tech + jobs_trans:
        job_skill_counts.update(skill for skill in skills if _is_valid_skill(skill))

    for skill, count in job_skill_counts.items():
        if count >= MIN_JOB_SKILL_SUPPORT_FOR_MODULES:
            known_skills.add(skill)

    return known_skills


def _module_skill_text(row: pd.Series) -> str:
    # Keep technical-skill evidence fairly tight: title + description only.
    return " ".join(
        part
        for part in [
            str(row.get("module_title", "")).strip(),
            str(row.get("module_description", "")).strip(),
        ]
        if part
    )


def _module_transferable_text(row: pd.Series) -> str:
    # This stays separate from `_module_skill_text` so transferable-skill cue extraction
    # can evolve independently later.
    return " ".join(
        part
        for part in [
            str(row.get("module_title", "")).strip(),
            str(row.get("module_description", "")).strip(),
        ]
        if part
    )


def _parse_workload_shares(workload: object) -> Dict[str, float]:
    cleaned = str(workload or "").strip().lower()
    if not cleaned:
        return {}

    totals: Dict[str, float] = {}
    for label, value in re.findall(
        r"\b(lecture|tutorial|lab|project|preparation)_([0-9]*\.?[0-9]+)\b",
        cleaned,
    ):
        totals[label] = totals.get(label, 0.0) + float(value)

    total_hours = sum(totals.values())
    if total_hours <= 0:
        return {}

    return {label: value / total_hours for label, value in totals.items()}


def _soft_skill_scores_from_terms(terms: Iterable[str]) -> Dict[str, float]:
    scores = {skill: 0.0 for skill in SOFT_SKILL_MARKERS}
    for raw_term in terms:
        term = _clean_token(str(raw_term))
        if not term:
            continue
        for soft_skill, markers in SOFT_SKILL_MARKERS.items():
            if any(marker in term for marker in markers):
                scores[soft_skill] = max(scores[soft_skill], 1.0)
    return scores


def _apply_workload_soft_skill_scores(scores: Dict[str, float], workload: object) -> None:
    workload_shares = _parse_workload_shares(workload)
    if not workload_shares:
        return

    # Workload contributes only weak soft-skill hints, never technical skill labels.
    for soft_skill, weights in WORKLOAD_SOFT_SKILL_WEIGHTS.items():
        contribution = sum(
            workload_shares.get(label, 0.0) * weight for label, weight in weights.items()
        )
        if contribution > 0:
            scores[soft_skill] += contribution


def _rank_soft_skills(scores: Dict[str, float]) -> List[str]:
    ranked = [
        (skill, score)
        for skill, score in scores.items()
        if score >= SOFT_SKILL_MIN_SCORE
    ]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [skill for skill, _ in ranked[:SOFT_SKILL_MAX_ITEMS]]


def apply_skill_taxonomy(
    config: PipelineConfig,
    role_rules: Dict[str, object],
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
) -> SkillTaxonomyResult:
    alias_map = load_skill_aliases(config.skill_aliases_file)
    skillsfuture_df = load_skillsfuture_mapping()
    channel_map = _build_channel_map(skillsfuture_df)

    transferable_cues = role_rules.get("transferable_cues", [])
    if not isinstance(transferable_cues, list):
        transferable_cues = []

    jobs_out = jobs.copy().drop(
        columns=["transferable_skills", "transferable_cues", "soft_skills"],
        errors="ignore",
    )
    modules_out = modules.copy().drop(
        columns=["transferable_skills", "transferable_cues", "soft_skills"],
        errors="ignore",
    )

    # Jobs arrive with explicit skill lists, so this pass mostly normalizes and splits
    # them into technical vs transferable / soft channels.
    jobs_tech  = []
    jobs_trans = []
    jobs_soft  = []

    unmapped_skills = set() 

    for _, row in jobs_out.iterrows():
        skills = row.get("skills", [])
        if not isinstance(skills, list):
            skills = []

        tech, trans, unmapped_job = _split_channels(skills, alias_map, channel_map)
        jobs_tech.append(tech)
        jobs_trans.append(trans)
        unmapped_skills.update(unmapped_job)

        text_blob = " ".join([str(row.get("title", "")), str(row.get("description_clean", ""))])
        soft_scores = _soft_skill_scores_from_terms(
            list(trans) + _extract_transferable_cues(text_blob, transferable_cues)
        )
        jobs_soft.append(_rank_soft_skills(soft_scores))

    jobs_out["technical_skills"] = jobs_tech
    jobs_out["soft_skills"] = jobs_soft

    # Modules do not have an explicit `skills` field, so build the vocabulary first and
    # then infer technical/soft skills from module text and workload.
    known_skills = _build_known_skills(channel_map, jobs_tech, jobs_trans)

    module_tech: List[List[str]] = []
    module_soft: List[List[str]] = []

    for _, row in modules_out.iterrows():
        skill_text = _module_skill_text(row)
        mentions = _extract_mentions(skill_text, known_skills)
        tech, trans, unmapped_module = _split_channels(mentions, alias_map, channel_map)
        unmapped_skills.update(unmapped_module)
        soft_scores = _soft_skill_scores_from_terms(
            list(trans) + _extract_transferable_cues(skill_text, transferable_cues)
        )
        _apply_workload_soft_skill_scores(soft_scores, row.get("module_workload", ""))

        module_tech.append(tech)
        module_soft.append(_rank_soft_skills(soft_scores))

    modules_out["technical_skills"] = module_tech
    modules_out["soft_skills"] = module_soft

    diagnostics: Dict[str, float | str] = {
        "skillsfuture_rows": float(len(skillsfuture_df)),
        "skill_channel_map_size": float(len(channel_map)),
        "jobs_with_technical_skills": _count_nonempty_lists(jobs_out["technical_skills"]),
        "jobs_with_soft_skills": _count_nonempty_lists(jobs_out["soft_skills"]),
        "modules_with_technical_skills": _count_nonempty_lists(
            modules_out["technical_skills"]
        ),
        "modules_with_soft_skills": _count_nonempty_lists(modules_out["soft_skills"]),
        "known_skill_vocab_size": float(len(known_skills)),
        "unmapped_skills_count": float(len(unmapped_skills)),
        "unmapped_skills_sample": ", ".join(sorted(list(unmapped_skills))[:10]) if unmapped_skills else "none",
    }

    return SkillTaxonomyResult(
        jobs=jobs_out,
        modules=modules_out,
        skill_channel_map=channel_map,
        diagnostics=diagnostics,
    )
