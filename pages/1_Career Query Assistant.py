from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.module_readiness.api import load_dashboard_query_backend
from src.module_readiness.llm import OpenAICompatibleLLMClient, explain_job_query


APP_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = APP_ROOT / "outputs"

FACULTY_FULL = {
    "BIZ": "Business",
    "CDE": "Design & Eng",
    "CDE/SOC": "CDE / Computing",
    "SOC": "Computing",
    "CHS": "Humanities & Sciences",
    "LAW": "Law",
    "MED": "Medicine",
    "DEN": "Dentistry",
    "YST": "Music (YST)",
}


def _fmt_tags(values: object) -> str:
    items = values if isinstance(values, list) else []
    clean = [str(item).strip() for item in items if str(item).strip()]
    return ", ".join(clean[:8]) if clean else "None highlighted"


def _skill_pills(skills: list[str], color: str = "#e8eaf6", text_color: str = "#283593", limit: int = 8) -> str:
    """Render a list of skills as inline HTML pill badges."""
    clean = [str(s).strip() for s in skills if str(s).strip()][:limit]
    if not clean:
        return '<span style="color:#999;font-size:0.85em">—</span>'
    pills = "".join(
        f'<span style="display:inline-block;background:{color};color:{text_color};'
        f'border-radius:12px;padding:2px 10px;margin:2px 3px;font-size:0.82em">{s}</span>'
        for s in clean
    )
    return pills


@st.cache_resource(show_spinner=False)
def _get_query_backend():
    return load_dashboard_query_backend(output_dir=OUTPUTS_DIR)


@st.cache_resource(show_spinner=False)
def _get_llm_client() -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient.from_env()


def _degree_filter_sidebar(summary: pd.DataFrame) -> tuple[str | None, str | None]:
    st.sidebar.markdown("## Query Settings")
    exp_max = st.sidebar.slider("Max experience (years)", min_value=0, max_value=5, value=2)
    top_jobs = st.sidebar.slider("Retrieved jobs", min_value=3, max_value=10, value=5)
    top_modules = st.sidebar.slider("Retrieved modules", min_value=3, max_value=12, value=8)
    st.session_state["job_query_exp_max"] = exp_max
    st.session_state["job_query_top_jobs"] = top_jobs
    st.session_state["job_query_top_modules"] = top_modules

    restrict_to_degree = st.sidebar.checkbox(
        "Restrict modules to one degree",
        value=False,
        help="When enabled, module recommendations are limited to required modules in the selected degree.",
    )
    if not restrict_to_degree or summary.empty:
        return None, None

    fac_opts = (
        summary[["faculty", "faculty_code"]]
        .drop_duplicates()
        .assign(
            label=lambda d: d.apply(
                lambda r: f"{r['faculty_code']} · {FACULTY_FULL.get(str(r['faculty_code']), r['faculty'])}",
                axis=1,
            )
        )
        .sort_values("faculty_code")
    )
    sel_fac_label = st.sidebar.selectbox("Faculty", fac_opts["label"].tolist(), key="job_query_faculty")
    sel_fac = str(fac_opts.loc[fac_opts["label"] == sel_fac_label, "faculty"].iloc[0])

    filtered = summary[summary["faculty"].astype(str) == sel_fac].sort_values("primary_major")
    programmes = filtered["primary_major"].astype(str).drop_duplicates().tolist()
    sel_programme = st.sidebar.selectbox("Programme", programmes, key="job_query_programme")
    row = filtered[filtered["primary_major"].astype(str) == sel_programme].iloc[0]
    return str(row["degree_id"]), str(row["primary_major"])


def _render_job_results(jobs: pd.DataFrame) -> None:
    st.markdown("#### Retrieved Jobs")
    if jobs.empty:
        st.warning("No jobs matched the query.")
        return

    for i, (_, row) in enumerate(jobs.iterrows()):
        title = str(row.get("title", ""))
        company = str(row.get("company", ""))
        role_family = str(row.get("role_family_name") or row.get("role_family", ""))
        tech_skills = row.get("technical_skills", [])
        if not isinstance(tech_skills, list):
            tech_skills = []
        soft_skills = row.get("soft_skills", [])
        if not isinstance(soft_skills, list):
            soft_skills = []
        summary = str(row.get("job_summary", ""))

        with st.expander(f"**{i + 1}. {title}**  —  {company}", expanded=(i == 0)):
            if role_family:
                st.markdown(
                    f'<span style="display:inline-block;background:#E3F2FD;color:#1565C0;'
                    f'border-radius:12px;padding:2px 10px;font-size:0.82em;margin-bottom:6px">'
                    f'{role_family}</span>',
                    unsafe_allow_html=True,
                )
            if summary:
                st.markdown(f'<div style="font-size:0.9em;color:#444;margin:6px 0 10px 0">{summary}</div>', unsafe_allow_html=True)
            if tech_skills:
                st.markdown(f'**Technical Skills** {_skill_pills(tech_skills, "#E8EAF6", "#283593")}', unsafe_allow_html=True)
            if soft_skills:
                st.markdown(f'**Soft Skills** {_skill_pills(soft_skills, "#F3E5F5", "#6A1B9A")}', unsafe_allow_html=True)


def _render_module_results(modules: pd.DataFrame, degree_label: str | None) -> None:
    st.markdown("#### Recommended Modules")
    if degree_label:
        st.caption(f"Restricted to required modules in **{degree_label}**.")
    if modules.empty:
        st.warning("No modules met the retrieval threshold for the selected evidence set.")
        return

    for i, (_, row) in enumerate(modules.iterrows()):
        code = str(row.get("module_code", ""))
        title = str(row.get("module_title", ""))
        summary = str(row.get("module_summary", ""))
        overlap = row.get("technical_skill_overlap", [])
        if not isinstance(overlap, list):
            overlap = []
        tech_skills = row.get("technical_skills", [])
        if not isinstance(tech_skills, list):
            tech_skills = []
        soft_skills = row.get("soft_skills", [])
        if not isinstance(soft_skills, list):
            soft_skills = []
        matched_count = int(row.get("matched_job_count", 0) or 0)

        with st.expander(f"**{code}** — {title}", expanded=(i == 0)):
            cols = st.columns([3, 1])
            with cols[0]:
                if summary:
                    st.markdown(f'<div style="font-size:0.9em;color:#444;margin-bottom:8px">{summary}</div>', unsafe_allow_html=True)
            with cols[1]:
                if matched_count:
                    st.metric("Matched Jobs", matched_count)
            if overlap:
                st.markdown(f'**Skill Overlap with Jobs** {_skill_pills(overlap, "#E8F5E9", "#1B5E20")}', unsafe_allow_html=True)
            if tech_skills:
                st.markdown(f'**Technical Skills** {_skill_pills(tech_skills, "#E8EAF6", "#283593")}', unsafe_allow_html=True)
            if soft_skills:
                st.markdown(f'**Soft Skills** {_skill_pills(soft_skills, "#F3E5F5", "#6A1B9A")}', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="Career Query Assistant — MOE",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    backend = _get_query_backend()
    client = _get_llm_client()
    degree_id, degree_label = _degree_filter_sidebar(backend.degree_summary)

    st.title("🎯 Career Query Assistant")
    st.caption(
        "Type a job or role in natural language. The system retrieves matching early-career job postings, "
        "finds relevant NUS modules based on the skills required, and explains the reasoning behind the recommendations. "
    )
    if client.configured:
        st.caption(f"LLM status: {client.provider_label} · model `{client.model}`")
    else:
        st.caption(
            "LLM status: not configured. The page will fall back to a deterministic summary until "
            "`LLM_API_KEY` or `OPENAI_API_KEY` is available."
        )

    with st.form("job_query_form", clear_on_submit=False):
        query = st.text_input(
            "Describe the jobs you want to search for",
            value=st.session_state.get("job_query_input", "Data scientist jobs"),
            placeholder="e.g. Data scientist jobs using Python, SQL, and machine learning",
        )
        st.caption(
            "Examples: `Data scientist jobs`, `Entry-level cybersecurity analyst roles`, "
            "`Business analyst jobs with SQL and dashboards`."
        )
        submitted = st.form_submit_button("Retrieve and Explain")

    if not submitted:
        return

    query = str(query).strip()
    st.session_state["job_query_input"] = query
    if not query:
        st.warning("Enter a job query first.")
        return

    with st.spinner("Retrieving jobs, ranking modules, and generating explanation..."):
        result = backend.run_job_query(
            natural_language_query=query,
            top_job_k=int(st.session_state.get("job_query_top_jobs", 5)),
            top_module_k=int(st.session_state.get("job_query_top_modules", 8)),
            exp_max=int(st.session_state.get("job_query_exp_max", 2)),
            degree_id=degree_id,
        )
        explanation = explain_job_query(
            natural_language_query=query,
            jobs=result.jobs,
            modules=result.modules,
            degree_label=result.degree_label,
            client=client,
        )

    st.markdown("---")
    st.markdown("### Analysis")
    if explanation.used_fallback:
        st.info("Showing deterministic summary because the LLM call was unavailable.")
        if explanation.error:
            st.caption(f"LLM error: {explanation.error}")
    else:
        st.caption(f"Generated with {explanation.provider_label} · `{explanation.model}`")
    st.markdown(explanation.markdown)

    st.markdown("---")
    st.markdown("### Retrieved Jobs & Recommended Modules")
    left, right = st.columns([1, 1], gap="large")
    with left:
        _render_job_results(result.jobs)
    with right:
        _render_module_results(result.modules, result.degree_label)


if __name__ == "__main__":
    main()
