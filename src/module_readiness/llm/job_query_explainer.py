from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def _truncate(text: object, limit: int = 220) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not value or str(value).strip() in {"", "nan"}:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def _top_terms(frame: pd.DataFrame, column: str, limit: int = 8) -> list[str]:
    counts: dict[str, int] = {}
    if column not in frame.columns:
        return []
    for value in frame[column].tolist():
        for item in _as_list(value):
            key = item.lower()
            counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _ in ordered[:limit]]


def build_job_query_prompt_context(
    natural_language_query: str,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    *,
    degree_label: str | None = None,
) -> str:
    job_records = []
    for _, row in jobs.head(5).iterrows():
        job_records.append(
            {
                "job_id": str(row.get("job_id", "")),
                "title": str(row.get("title", "")),
                "company": str(row.get("company", "")),
                "role_family": str(row.get("role_family_name") or row.get("role_family", "")),
                "primary_category": str(row.get("primary_category", "")),
                "hybrid_score": round(float(row.get("score", 0.0)), 4),
                "technical_skills": _as_list(row.get("technical_skills", []))[:8],
                "soft_skills": _as_list(row.get("soft_skills", []))[:6],
                "summary": _truncate(row.get("job_summary", row.get("description_clean", ""))),
                "evidence_terms": str(row.get("evidence_terms", "")),
            }
        )

    module_records = []
    for _, row in modules.head(8).iterrows():
        module_records.append(
            {
                "module_code": str(row.get("module_code", "")),
                "module_title": str(row.get("module_title", "")),
                "top_role_family": str(row.get("top_role_family_name") or row.get("top_role_family", "")),
                "hybrid_score": round(float(row.get("similarity_score", 0.0)), 4),
                "matched_job_count": int(row.get("matched_job_count", 0) or 0),
                "matched_jobs": str(row.get("matched_jobs", "")),
                "technical_skill_overlap": _as_list(row.get("technical_skill_overlap", []))[:6],
                "soft_skill_overlap": _as_list(row.get("soft_skill_overlap", []))[:4],
                "technical_skills": _as_list(row.get("technical_skills", []))[:8],
                "summary": _truncate(row.get("module_summary", row.get("module_description", ""))),
                "evidence_terms": str(row.get("job_evidence_terms") or row.get("evidence_terms", "")),
            }
        )

    payload = {
        "query": natural_language_query,
        "degree_context": degree_label or "",
        "retrieved_jobs": job_records,
        "recommended_modules": module_records,
    }
    return json.dumps(payload, indent=2)


def build_fallback_job_query_explanation(
    natural_language_query: str,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    *,
    degree_label: str | None = None,
) -> str:
    if jobs.empty:
        return (
            f"No matching early-career jobs were retrieved for `{natural_language_query}`. "
            "Try a broader description or add a few concrete skills."
        )

    top_role_families = _top_terms(jobs, "role_family_name", limit=3)
    recurring_job_skills = _top_terms(jobs, "technical_skills", limit=6)
    recurring_module_skills = _top_terms(modules, "technical_skill_overlap", limit=6)

    overview = (
        f"The query `{natural_language_query}` retrieved jobs that mainly cluster around "
        f"{', '.join(top_role_families) if top_role_families else 'a small set of related roles'}."
    )
    if degree_label:
        overview += f" Module recommendations were restricted to the required curriculum for `{degree_label}`."

    job_line = (
        "Across the retrieved postings, the most repeated technical signals are "
        f"{', '.join(recurring_job_skills) if recurring_job_skills else 'broad domain terms rather than a tight skill cluster'}."
    )

    if modules.empty:
        module_line = (
            "No modules cleared the retrieval threshold for the selected evidence set, so the query likely needs to be broadened "
            "or the degree filter relaxed."
        )
    else:
        top_modules = [
            f"`{row['module_code']}` {row['module_title']}"
            for _, row in modules.head(3).iterrows()
        ]
        overlap_text = (
            f" The strongest shared module-job skills are {', '.join(recurring_module_skills)}."
            if recurring_module_skills
            else ""
        )
        module_line = (
            "The strongest module matches are "
            f"{', '.join(top_modules)}. These modules align because they repeatedly match the retrieved jobs on content and skill signals."
            f"{overlap_text}"
        )

    caution = (
        "Treat this as retrieval evidence rather than a causal placement claim: the page surfaces jobs and modules with similar language "
        "and skills, not guaranteed graduate outcomes."
    )

    return "\n\n".join([overview, job_line, module_line, caution])


@dataclass
class JobQueryExplanation:
    markdown: str
    used_fallback: bool
    provider_label: str
    model: str | None = None
    error: str | None = None


@dataclass
class OpenAICompatibleLLMClient:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 30

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLMClient":
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        base_url = (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        model = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
        timeout_seconds = int(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
        return cls(
            api_key=str(api_key).strip(),
            base_url=str(base_url).strip(),
            model=str(model).strip(),
            timeout_seconds=timeout_seconds,
        )

    @property
    def provider_label(self) -> str:
        return "OpenAI-compatible API"

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

    def generate_markdown(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        if not self.configured:
            raise RuntimeError("LLM credentials are not configured.")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM network error: {exc}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response did not contain any choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM response did not contain text content.")
        return content.strip()


def explain_job_query(
    natural_language_query: str,
    jobs: pd.DataFrame,
    modules: pd.DataFrame,
    *,
    degree_label: str | None = None,
    client: OpenAICompatibleLLMClient | None = None,
) -> JobQueryExplanation:
    fallback = build_fallback_job_query_explanation(
        natural_language_query=natural_language_query,
        jobs=jobs,
        modules=modules,
        degree_label=degree_label,
    )
    client = client or OpenAICompatibleLLMClient.from_env()
    if not client.configured:
        return JobQueryExplanation(
            markdown=fallback,
            used_fallback=True,
            provider_label=client.provider_label,
            model=client.model or None,
            error="LLM_API_KEY or OPENAI_API_KEY is not configured.",
        )

    system_prompt = (
        "You are assisting MOE officers using a curriculum-readiness analytics tool. "
        "Summarize only what is supported by the retrieved evidence. "
        "Do not invent jobs, modules, skills, or claims. "
        "Write concise markdown with: "
        "1 short overview paragraph, "
        "3 bullet points on retrieved jobs and recurring skills, "
        "3 bullet points on relevant modules and why they fit, "
        "and 1 short caution sentence."
    )
    user_prompt = (
        "Use the following JSON evidence from the retrieval system.\n\n"
        f"```json\n{build_job_query_prompt_context(natural_language_query, jobs, modules, degree_label=degree_label)}\n```"
    )

    try:
        markdown = client.generate_markdown(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return JobQueryExplanation(
            markdown=markdown,
            used_fallback=False,
            provider_label=client.provider_label,
            model=client.model,
        )
    except Exception as exc:
        return JobQueryExplanation(
            markdown=fallback,
            used_fallback=True,
            provider_label=client.provider_label,
            model=client.model,
            error=str(exc),
        )
