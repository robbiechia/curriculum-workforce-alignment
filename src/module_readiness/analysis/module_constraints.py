"""
Module prerequisite and preclusion constraint engine.

Builds a prerequisite DAG and preclusion set once at load time, then provides
fast per-degree reachability and preclusion filtering for elective gating.

Design
------
Prerequisites are modelled as a list of OR-groups per module:
    prereqs[M] = [[A, B], [C, D, E]]   # must satisfy group 0 AND group 1
    group 0: need at least one of {A, B}
    group 1: need at least one of {C, D, E}

Reachability (fixpoint BFS)
    seed = degree's required module set
    expand: if ALL OR-groups of M are satisfiable from the current reachable
            set → M becomes reachable
    iterate until stable (convergence guaranteed because reachable only grows)

Wildcards
    prereq codes like ACC1701% are expanded to all known modules matching
    the prefix at construction time. This is done once.

Preclusions
    If a degree's required curriculum contains a module that precludes X,
    then X is not available as an elective for that degree.

Complexity
----------
Construction: O(pairs) — one pass over the CSV rows
Per-degree reachability:
    worst-case O(modules × max_and_depth × max_or_width) per fixpoint pass
    empirically: ~7k modules, depth ≤ 3, converges in ≤ 6 passes
    → ~126k set-membership checks per degree, negligible

Safety valve
    If fixpoint does not converge within MAX_FIXPOINT_ITERATIONS, a warning
    is emitted and expansion stops. This guards against any circular prereq
    data in the source.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

MAX_FIXPOINT_ITERATIONS = 20
# Warn (not raise) if the total ops across ALL fixpoint iterations for one degree
# exceeds this. With ~3k modules that have prereqs and ~13 max AND-depth,
# a single iteration is ~8-9k ops; 6 iterations ~ 50-60k is normal.
# Flag anything beyond 500k as unexpectedly expensive.
EXPANSION_WARN_THRESHOLD = 500_000


class ModuleConstraints:
    """
    Immutable constraint engine built from prereq and preclusion CSVs.
    Construct once per pipeline run; reuse across all degrees.
    """

    def __init__(
        self,
        prereqs_path: Path | str,
        preclusions_path: Path | str,
    ) -> None:
        prereqs_path = Path(prereqs_path)
        preclusions_path = Path(preclusions_path)

        self._all_codes: set[str] = set()
        # prereqs[module_code] = list of OR-groups, each group is a frozenset of codes
        self._prereqs: dict[str, list[frozenset[str]]] = {}
        # precludes[module_code] = set of codes that this module precludes
        self._precludes: dict[str, set[str]] = {}
        # precluded_by[code] = set of modules whose presence precludes this code
        self._precluded_by: dict[str, set[str]] = {}

        if prereqs_path.exists():
            self._load_prereqs(pd.read_csv(prereqs_path, keep_default_na=False))
        else:
            warnings.warn(f"[ModuleConstraints] prereqs file not found: {prereqs_path}", stacklevel=2)

        if preclusions_path.exists():
            self._load_preclusions(pd.read_csv(preclusions_path, keep_default_na=False))
        else:
            warnings.warn(f"[ModuleConstraints] preclusions file not found: {preclusions_path}", stacklevel=2)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_prereqs(self, df: pd.DataFrame) -> None:
        # Collect full module universe first (needed for wildcard expansion)
        self._all_codes.update(df["module_code"].str.strip().str.upper().unique())
        self._all_codes.update(df["prereq_code"].str.strip().str.upper().unique())

        # Group by (module_code, or_group) to build OR-group frozensets,
        # then collect groups per module code.
        tmp: dict[str, dict[int, set[str]]] = {}
        for _, row in df.iterrows():
            code = str(row["module_code"]).strip().upper()
            prereq = str(row["prereq_code"]).strip().upper()
            group_idx = int(row["or_group"])
            has_wildcard = str(row.get("has_wildcard", "False")).lower() in ("true", "1", "yes")

            tmp.setdefault(code, {}).setdefault(group_idx, set())

            if has_wildcard or "%" in prereq:
                stem = prereq.rstrip("%")
                expanded = {c for c in self._all_codes if c.startswith(stem)}
                if expanded:
                    tmp[code][group_idx].update(expanded)
                else:
                    # Wildcard matched nothing — keep raw token so it never
                    # accidentally satisfies the group.
                    tmp[code][group_idx].add(prereq)
            else:
                tmp[code][group_idx].add(prereq)

        for code, groups in tmp.items():
            self._prereqs[code] = [frozenset(g) for g in groups.values()]

    def _load_preclusions(self, df: pd.DataFrame) -> None:
        self._all_codes.update(df["module_code"].str.strip().str.upper().unique())
        self._all_codes.update(df["precluded_code"].str.strip().str.upper().unique())

        for _, row in df.iterrows():
            code = str(row["module_code"]).strip().upper()
            precluded = str(row["precluded_code"]).strip().upper()
            has_wildcard = str(row.get("has_wildcard", "False")).lower() in ("true", "1", "yes")

            if has_wildcard or "%" in precluded:
                stem = precluded.rstrip("%")
                targets = {c for c in self._all_codes if c.startswith(stem)} or {precluded}
            else:
                targets = {precluded}

            self._precludes.setdefault(code, set()).update(targets)
            for t in targets:
                self._precluded_by.setdefault(t, set()).add(code)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reachable_from(self, degree_modules: Iterable[str]) -> set[str]:
        """
        Return all module codes reachable (prereqs transitively satisfied)
        given the set of modules already in the degree curriculum.

        The input set itself is always in the returned reachable set.
        """
        reachable: set[str] = {c.strip().upper() for c in degree_modules}
        total_ops = 0

        for iteration in range(MAX_FIXPOINT_ITERATIONS):
            newly_added: set[str] = set()

            for code, groups in self._prereqs.items():
                if code in reachable:
                    continue
                # A module is unlockable if every AND-group has at least one
                # member already in the reachable set.
                total_ops += sum(len(g) for g in groups)
                if all(any(m in reachable for m in group) for group in groups):
                    newly_added.add(code)

            reachable |= newly_added
            if not newly_added:
                break  # fixpoint reached
        else:
            warnings.warn(
                f"[ModuleConstraints] fixpoint did not converge after {MAX_FIXPOINT_ITERATIONS} "
                "iterations — possible circular prerequisites in source data. "
                "Returning partial reachable set.",
                stacklevel=2,
            )

        if total_ops > EXPANSION_WARN_THRESHOLD:
            warnings.warn(
                f"[ModuleConstraints] reachability expansion used {total_ops:,} operations "
                f"({iteration + 1} iterations). Consider caching or reducing corpus size.",
                stacklevel=2,
            )

        return reachable

    def precluded_by_curriculum(self, degree_modules: Iterable[str]) -> set[str]:
        """
        Return all module codes precluded because the degree curriculum
        contains a module that precludes them.
        """
        precluded: set[str] = set()
        for code in degree_modules:
            c = code.strip().upper()
            precluded.update(self._precludes.get(c, set()))
        return precluded

    def filter_electives(
        self,
        degree_modules: Iterable[str],
        candidates: Iterable[str],
    ) -> set[str]:
        """
        Given the degree's required module set and a pool of elective candidates,
        return the subset of candidates that are:
          (a) prereq-reachable from the degree curriculum, AND
          (b) not precluded by any module in the degree curriculum.

        This is the main entry point for elective gating.
        """
        degree_set = {c.strip().upper() for c in degree_modules}
        candidate_set = {c.strip().upper() for c in candidates}

        reachable = self.reachable_from(degree_set)
        precluded = self.precluded_by_curriculum(degree_set)

        # Modules already in the required curriculum are never elective candidates
        return (candidate_set & reachable) - precluded - degree_set
