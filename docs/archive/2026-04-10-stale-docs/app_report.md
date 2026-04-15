# Application Report (Test2): Query Interfaces and User-Facing APIs

Generated on: 2026-03-26

## Overview
This report documents the user-facing query interfaces and application programming interfaces (APIs) available for interacting with the module readiness pipeline. These interfaces are separate from the core pipeline scoring logic and are designed for exploratory analysis and decision support.

## 1) Query interfaces and retrieval behavior

The system provides three main query interfaces for exploration and module-to-job matching:

### 1.1 Search Jobs by Natural Language Query
**`search_jobs(natural_language_query, exp_max=2, top_k=10)`**

This interface allows users to describe a job or role in natural language and retrieve matching job advertisements from the database.

**Input:**
- Free-text query (e.g., "I want to work in machine learning engineering")
- Optional maximum years of experience filter (exp_max)
- Optional top-K results limit

**Output:**
- Ranked list of job postings with similarity scores
- Breakdown of evidence (matching terms and skills)

**Scoring approach:** Currently uses a fixed lexical/skill blend (75% description similarity, 25% skill matching) for ranking.

### 1.2 Recommend Modules for Selected Jobs
**`recommend_relevant_modules(query_or_job_ids, top_k=10, role_family=None)`**

This interface identifies NUS modules that best prepare students for selected job(s). Users can input either:
- Job IDs (if they've selected specific postings)
- Natural language query (system will match to jobs first)

**Input:**
- Job identifier(s) or free-text query
- Optional role family filter (e.g., "Software Development")
- Optional top-K modules limit

**Output:**
- Ranked list of modules with alignment scores
- Technical and transferable skill alignment details
- Related role families and evidence

**Scoring approach:** Computes module-job similarity using the full pipeline (lexical + technical skill + transferable signals) and aggregates across selected job(s).

### 1.3 Get Module Role Profile
**`get_module_role_profile(module_code, top_families=5)`**

This interface provides a module-centric view showing how a module aligns with different occupational roles in the labor market.

**Input:**
- NUS module code (e.g., "CS3244")
- Optional top-K role families to display

**Output:**
- Top role families this module prepares for
- Alignment scores for each role
- Distribution of job demand across roles
- Evidence job count for each role

**Scoring approach:** Retrieves pre-computed module-role aggregation scores from the pipeline, ranked by role fit.

## 2) Use cases

**Curriculum Planning:** Administrators can use these interfaces to explore which modules support strategic hiring needs.

**Student Advising:** Advisors can show students which modules align best with their career interests.

**Gap Analysis:** Identify which occupational roles are underserved by current module offerings.

**Evidence Review:** Inspect which specific jobs informed the module-role alignment scores.
