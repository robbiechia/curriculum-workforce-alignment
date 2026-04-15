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


@st.cache_resource(show_spinner=False)
def _get_query_backend():
    return load_dashboard_query_backend(output_dir=OUTPUTS_DIR)


@st.cache_resource(show_spinner=False)
def _get_llm_client() -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient.from_env()


def _degree_filter_sidebar(summary: pd.DataFrame) -> tuple[str | None, str | None]:
    st.sidebar.markdown("## Assistant Settings")
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
    st.subheader("Top Retrieved Jobs")
    if jobs.empty:
        st.warning("No jobs matched the query.")
        return

    view = jobs[["title", "company"]].copy()
    view = view.rename(
        columns={
            "title": "Job Title",
            "company": "Company",
        }
    )
    st.dataframe(view, width="stretch", hide_index=True)

    for _, row in jobs.iterrows():
        with st.expander(f"{row['title']} · {row['company']}", expanded=False):
            st.write(row.get("job_summary", ""))
            st.caption(f"Technical skills: {_fmt_tags(row.get('technical_skills', []))}")
            st.caption(f"Soft skills: {_fmt_tags(row.get('soft_skills', []))}")


def _render_module_results(modules: pd.DataFrame, degree_label: str | None) -> None:
    st.subheader("Recommended Modules")
    if degree_label:
        st.caption(f"Recommendations are restricted to required modules in `{degree_label}`.")
    if modules.empty:
        st.warning("No modules met the retrieval threshold for the selected evidence set.")
        return

    view = modules[["module_code", "module_title"]].copy()
    view = view.rename(
        columns={
            "module_code": "Module Code",
            "module_title": "Module Title",
        }
    )
    st.dataframe(view, width="stretch", hide_index=True)

    for _, row in modules.iterrows():
        title = f"{row['module_code']} · {row['module_title']}"
        with st.expander(title, expanded=False):
            st.write(row.get("module_summary", ""))
            st.caption(f"Technical skill overlap: {_fmt_tags(row.get('technical_skill_overlap', []))}")
            st.caption(f"Technical skills: {_fmt_tags(row.get('technical_skills', []))}")
            st.caption(f"Soft skills: {_fmt_tags(row.get('soft_skills', []))}")


def main() -> None:
    st.set_page_config(
        page_title="Job Query Assistant — MOE",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    backend = _get_query_backend()
    client = _get_llm_client()
    degree_id, degree_label = _degree_filter_sidebar(backend.degree_summary)

    st.title("Natural-Language Job Assistant")
    st.caption(
        "Type a job or role in natural language. The system retrieves matching early-career job postings, "
        "finds relevant NUS modules, and generates an officer-facing explanation grounded in those results."
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

    st.subheader("Explanation")
    if explanation.used_fallback:
        st.info("Showing deterministic summary because the LLM call was unavailable.")
        if explanation.error:
            st.caption(f"LLM error: {explanation.error}")
    else:
        st.caption(f"Generated with {explanation.provider_label} · `{explanation.model}`")
    st.markdown(explanation.markdown)

    left, right = st.columns([1.05, 1.15], gap="large")
    with left:
        _render_job_results(result.jobs)
    with right:
        _render_module_results(result.modules, result.degree_label)


if __name__ == "__main__":
    main()
