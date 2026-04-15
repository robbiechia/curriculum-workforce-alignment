# Technical Report (Test2)

## Objective
Measure module-to-job readiness for early-career university-level jobs and provide explainable outputs for policy and curriculum use.

## Final method choices
- Scope filter: minimumYearsExperience <= 2 and ssecEqa = 70.
- Role grouping: jobs keep SSOC-5 and SSOC-4 codes for auditability, but final reporting uses curated role clusters.
- Skills governance: SkillsFuture Skills Framework mapping for technical skills, plus broad soft-skill inference from workload.
- Fit computation: hybrid retrieval over title/description text plus extracted technical skills using BM25 and sentence embeddings.
- Rank fusion: reciprocal rank fusion (RRF) combines BM25 and embedding rankings into one score.
- Aggregation: top-K module-job matches are averaged within curated role clusters and support-adjusted by evidence count.

## Formula summary
- retrieval_text = normalized(title/description text + extracted technical skills)
- RRFScore = RRF(rank_BM25(retrieval_text), rank_Embedding(retrieval_text))
- SSOC5RawScore(module, ssoc5) = mean_topK(RRFScore, K)
- SupportWeight(n) = n / (n + role_support_prior)
- RoleClusterScore(module, cluster) = mean_topK(RRFScore within cluster, K) * SupportWeight(evidence_job_count)

## End-to-end workflow
1. Read source tables from PostgreSQL (`raw_jobs`, `raw_modules`, `skillsfuture_mapping`, `ssoc2024_definitions`).
2. Filter jobs to early-career, degree-level scope (`minimumYearsExperience <= 2`, `ssecEqa = 70`).
3. Assign each job to a curated role cluster using SSOC plus title/skill split rules.
4. Prepare modules from NUSMods data and build module text from title/description/workload metadata.
5. Normalize skills with alias rules + SkillsFuture mapping and split into technical vs transferable signals.
6. Consolidate module variants (for example, suffix variants such as `ACC1701A/B`) into base module codes.
7. Build retrieval text for jobs and modules, then compute BM25 and sentence-embedding similarities.
8. Fuse BM25 and embedding rankings with Reciprocal Rank Fusion (RRF) to get module-job fit evidence.
9. Aggregate top-K evidence into module-level scores by SSOC and curated role clusters with support weighting.
10. Compute demand-supply skill gaps per role family and generate final outputs/reports.

## Runtime diagnostics
```json
{'jobs_raw_count': 22720.0, 'jobs_after_filters': 1994.0, 'jobs_filter_exp_max': 2.0, 'jobs_filter_ssec_eqa': '70', 'jobs_unique_ids': 1994.0, 'jobs_unique_ssoc': 280.0, 'jobs_unique_categories': 39.0, 'role_family_unique': 22.0, 'broad_family_unique': 8.0, 'role_family_ssoc_rows': 1376.0, 'role_family_keyword_rows': 502.0, 'role_family_category_rows': 0.0, 'role_family_fallback_rows': 116.0, 'ssoc_role_family_unique': 162.0, 'jobs_with_ssoc5_rows': 1994.0, 'jobs_with_ssoc4_rows': 1994.0, 'jobs_unique_ssoc5': 280.0, 'jobs_unique_ssoc4': 162.0, 'jobs_unique_ssoc5_names': 280.0, 'jobs_unique_ssoc4_names': 162.0, 'role_family_matched_keyword_rows': 490.0, 'ssoc4_official_map_size': 418.0, 'ssoc5_official_map_size': 1001.0, 'modules_rows': 4176.0, 'modules_catalog_year': '2024-2025', 'modules_catalog_source': 'db', 'modules_catalog_size': 7016.0, 'modules_selected_before_detail': 4176.0, 'modules_detail_available_rows': 4176.0, 'modules_unique_prefixes': 201.0, 'skillsfuture_rows': 2364.0, 'skill_channel_map_size': 2364.0, 'jobs_with_technical_skills': 1994.0, 'jobs_with_soft_skills': 1944.0, 'modules_with_technical_skills': 3660.0, 'modules_with_soft_skills': 3980.0, 'known_skill_vocab_size': 4222.0, 'unmapped_skills_count': 3184.0, 'unmapped_skills_sample': '3d, 3d modeling, 3d rendering, 3d studio max, 3gpp, 3pl, 5s, a+, ab testing, ability to learn', 'retrieval_jobs_rows': 1994.0, 'retrieval_modules_rows': 3752.0, 'retrieval_embedding_backend': 'sentence-transformers', 'retrieval_embedding_dimension': 384.0, 'retrieval_job_token_count_mean': 221.21313941825477, 'retrieval_module_token_count_mean': 65.41151385927505, 'scores_module_job_rows': 750400.0, 'scores_module_ssoc5_rows': 272548.0, 'scores_module_role_rows': 73526.0, 'scores_ranker': 'bm25+embeddings+rrf', 'indicators_module_rows': 3752.0, 'indicators_role_rows': 73526.0, 'indicators_gap_rows': 330.0, 'degree_source_kind': 'degree_plan', 'degree_plan_file': '/Users/arthurchong/Desktop/year_4/dsa4264/DSA4264-text-group4/data/nus_degree_plan.csv', 'degree_mapping_file': '/Users/arthurchong/Desktop/year_4/dsa4264/DSA4264-text-group4/degree_mapping/degree_mapping_AY2425.csv', 'degrees_rows': 73.0, 'degree_requirement_bucket_rows': 1027.0, 'degree_module_map_rows': 305604.0, 'degree_role_rows': 0.0, 'degree_ssoc5_rows': 0.0, 'degree_skill_supply_rows': 12113.0, 'degree_role_skill_gap_rows': 0.0, 'degree_ssoc5_skill_gap_rows': 0.0, 'degree_summary_rows': 73.0, 'degree_plan_expansion_audit_rows': 3935.0, 'degree_required_module_match_share': 0.9061268831559797, 'degree_unrestricted_module_map_rows': 294863.0, 'module_preclusions_path': '/Users/arthurchong/Desktop/year_4/dsa4264/DSA4264-text-group4/outputs/module_preclusions.csv', 'module_preclusions_generated': 'no', 'jobs_clean_written_path': '/Users/arthurchong/Desktop/year_4/dsa4264/DSA4264-text-group4/outputs/jobs_clean.parquet', 'modules_clean_written_path': '/Users/arthurchong/Desktop/year_4/dsa4264/DSA4264-text-group4/outputs/modules_clean.parquet', 'job_role_map_written_path': '/Users/arthurchong/Desktop/year_4/dsa4264/DSA4264-text-group4/outputs/job_role_map.parquet', 'jobs_clean_parquet_written': 'yes', 'modules_clean_parquet_written': 'yes', 'job_role_map_parquet_written': 'yes', 'db_outputs_persisted': 'no', 'db_degree_outputs_persisted': 'no', 'db_output_tables_count': 0.0}
```

## Default configuration
- top_k: 50
- bm25_k1: 1.2
- bm25_b: 0.75
- rrf_k: 60
- retrieval_top_n: 200
- bm25_min_score: 20
- bm25_relative_min: 0.25
- embedding_min_similarity: 0
- embedding_relative_min: 0
- role_support_prior: 5.0
- embedding_model_name: sentence-transformers/all-MiniLM-L6-v2