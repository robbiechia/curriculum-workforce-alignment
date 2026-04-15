"""Compatibility package for repo-root imports.

Some entry points import ``data_utils`` as a top-level package, while the actual
implementation lives under ``src/data_utils``. This shim keeps both launch modes
working without requiring callers to modify ``PYTHONPATH``.
"""
