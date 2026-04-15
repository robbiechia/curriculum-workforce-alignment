from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import certifi

SRC_PATH = Path(__file__).resolve().parents[1]
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

API_BASE_URL = "https://api.nusmods.com/v2"
DEFAULT_YEAR = "2024-2025"
DEFAULT_USER_AGENT = "module-readiness-nusmods-scraper"

# Fields to exclude when saving raw module detail JSON.
_EXCLUDED_DETAIL_FIELDS = {"semesterData"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _raw_year_dir(project_root: Path, year: str) -> Path:
    return project_root / "data" / "nusmods" / year


def _module_list_path(project_root: Path, year: str) -> Path:
    return _raw_year_dir(project_root, year) / "moduleList.json"


def _module_detail_path(project_root: Path, year: str, code: str) -> Path:
    return _raw_year_dir(project_root, year) / "modules" / f"{code}.json"


def _manifest_path(project_root: Path, year: str) -> Path:
    return _raw_year_dir(project_root, year) / "manifest.json"


def _read_json(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _http_get_json(url: str, timeout: int) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
        return json.loads(response.read().decode("utf-8"))


def _module_list_url(year: str) -> str:
    return f"{API_BASE_URL}/{year}/moduleList.json"


def _module_detail_url(year: str, code: str) -> str:
    return f"{API_BASE_URL}/{year}/modules/{code}.json"


def _load_raw_module_list(project_root: Path, year: str) -> Optional[List[Dict[str, object]]]:
    payload = _read_json(_module_list_path(project_root, year))
    return payload if isinstance(payload, list) else None


def _fetch_module_list(
    project_root: Path, year: str, timeout: int, force_refresh: bool
) -> Tuple[List[Dict[str, object]], str]:
    if not force_refresh:
        # Prefer the raw cache if it already exists so repeated setup runs stay cheap.
        cached = _load_raw_module_list(project_root, year)
        if cached is not None:
            return cached, "raw"

    payload = _http_get_json(_module_list_url(year), timeout=timeout)
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected module list payload for {year}")
    _write_json(_module_list_path(project_root, year), payload)
    return payload, "api"


def _load_raw_module_detail(
    project_root: Path, year: str, code: str
) -> Optional[Dict[str, object]]:
    payload = _read_json(_module_detail_path(project_root, year, code))
    return payload if isinstance(payload, dict) else None


def _fetch_module_detail(
    project_root: Path,
    year: str,
    code: str,
    timeout: int,
    force_refresh: bool,
) -> Tuple[str, Optional[Dict[str, object]], str]:
    if not force_refresh:
        cached = _load_raw_module_detail(project_root, year, code)
        if cached is not None:
            return code, cached, "raw"

    try:
        payload = _http_get_json(_module_detail_url(year, code), timeout=timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return code, None, "missing"

    if not isinstance(payload, dict):
        return code, None, "missing"

    # Strip excluded fields before persisting to disk.
    filtered = {k: v for k, v in payload.items() if k not in _EXCLUDED_DETAIL_FIELDS}
    _write_json(_module_detail_path(project_root, year, code), filtered)
    return code, filtered, "api"


def _collect_module_details(
    project_root: Path,
    year: str,
    module_codes: List[str],
    timeout: int,
    workers: int,
    force_refresh: bool,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, str], Dict[str, int]]:
    details: Dict[str, Dict[str, object]] = {}
    detail_sources: Dict[str, str] = {}
    stats: Dict[str, int] = {"api": 0, "raw": 0, "missing": 0}

    max_workers = max(1, workers)
    # Fetch per-module details concurrently because the NUSMods detail endpoint is the
    # slowest part of data setup.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _fetch_module_detail,
                project_root,
                year,
                code,
                timeout,
                force_refresh,
            )
            for code in module_codes
        ]
        for future in as_completed(futures):
            code, payload, source = future.result()
            stats[source] = stats.get(source, 0) + 1
            if payload is not None:
                details[code] = payload
                detail_sources[code] = source

    return details, detail_sources, stats


def _write_manifest(
    project_root: Path,
    year: str,
    module_count: int,
    detail_stats: Dict[str, int],
    catalog_source: str,
    force_refresh: bool,
) -> None:
    payload = {
        "academic_year": year,
        "api_base_url": API_BASE_URL,
        "catalog_endpoint": _module_list_url(year),
        "module_detail_endpoint_template": f"{API_BASE_URL}/{year}/modules/{{moduleCode}}.json",
        "module_count": module_count,
        "catalog_source": catalog_source,
        "detail_api_count": detail_stats.get("api", 0),
        "detail_raw_count": detail_stats.get("raw", 0),
        "detail_missing_count": detail_stats.get("missing", 0),
        "force_refresh": force_refresh,
        "excluded_fields": sorted(_EXCLUDED_DETAIL_FIELDS),
    }
    _write_json(_manifest_path(project_root, year), payload)


def scrape_modules(
    academic_year: str = DEFAULT_YEAR,
    project_root: Path | None = None,
    timeout: int = 20,
    workers: int = 16,
    force_refresh: bool = False,
) -> Dict[str, object]:
    root = project_root or _project_root()
    module_list, catalog_source = _fetch_module_list(root, academic_year, timeout, force_refresh)

    module_codes = sorted(
        {
            str(item.get("moduleCode") or "").strip().upper()
            for item in module_list
            if str(item.get("moduleCode") or "").strip()
        }
    )
    details, _, detail_stats = _collect_module_details(
        project_root=root,
        year=academic_year,
        module_codes=module_codes,
        timeout=timeout,
        workers=workers,
        force_refresh=force_refresh,
    )
    _write_manifest(
        project_root=root,
        year=academic_year,
        module_count=len(module_list),
        detail_stats=detail_stats,
        catalog_source=catalog_source,
        force_refresh=force_refresh,
    )

    return {
        "academic_year": academic_year,
        "catalog_source": catalog_source,
        "module_count": len(module_list),
        "detail_downloaded": detail_stats.get("api", 0),
        "detail_reused_from_raw": detail_stats.get("raw", 0),
        "detail_missing": detail_stats.get("missing", 0),
        "raw_dir": str(_raw_year_dir(root, academic_year)),
        "detail_available": len(details),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape raw NUSMods module data to data/raw/")
    parser.add_argument("--academic-year", default=DEFAULT_YEAR, help="Academic year, e.g. 2024-2025")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent module detail fetches")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Refetch catalog and module details even if raw JSON already exists",
    )
    args = parser.parse_args()

    summary = scrape_modules(
        academic_year=args.academic_year,
        timeout=args.timeout,
        workers=args.workers,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
