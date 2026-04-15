#!/usr/bin/env python3
from __future__ import annotations

import argparse
import site
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _site_packages_dir() -> Path:
    candidates = [Path(path) for path in site.getsitepackages()]
    if not candidates:
        raise RuntimeError("No site-packages directory found for the current Python interpreter.")
    return candidates[0]


def install_src_path() -> Path:
    repo_root = _repo_root()
    src_dir = repo_root / "src"
    if not src_dir.is_dir():
        raise RuntimeError(f"Expected src directory at {src_dir}, but it was not found.")

    site_packages = _site_packages_dir()
    pth_path = site_packages / "project_src.pth"
    pth_path.write_text(f"{src_dir}\n", encoding="utf-8")
    return pth_path


def remove_src_path() -> Path:
    pth_path = _site_packages_dir() / "project_src.pth"
    if pth_path.exists():
        pth_path.unlink()
    return pth_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install or remove a .pth file so the repository src/ directory is always on sys.path."
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove the project_src.pth file from the current interpreter's site-packages.",
    )
    args = parser.parse_args()

    if args.remove:
        path = remove_src_path()
        print(f"Removed {path} (if it existed)")
        return 0

    path = install_src_path()
    print(f"Using Python: {sys.executable}")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
