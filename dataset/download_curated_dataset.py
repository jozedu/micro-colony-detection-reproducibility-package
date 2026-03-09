#!/usr/bin/env python3
"""
download_curated_dataset.py
===========================
Download and extract curated dataset archives from Zenodo.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

ZENODO_DOI_URL = "https://doi.org/10.5281/zenodo.18505210"
ZENODO_RECORD_ID = "18703636"
ZENODO_PACKAGES = {
    "coco": {
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/micro_colonies_coco.zip/content",
        "archive_name": "micro_colonies_coco.zip",
        "sha256": "2c9762e8d89234b4dcd9c38a37b2d6478e4356ec4317c1f8d48b1ba7391bcef5",
    },
    "yolo": {
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/micro_colonies_yolov8.zip/content",
        "archive_name": "micro_colonies_yolov8.zip",
        "sha256": "210a53c87309ad2eb4937b3c9e446011eac7c3473f38817759c52fa75eb74ca3",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract curated dataset archive.",
    )
    parser.add_argument(
        "--package",
        choices=sorted(ZENODO_PACKAGES.keys()),
        help="Preset package on Zenodo: coco or yolo.",
    )
    parser.add_argument(
        "--url",
        help="Direct archive URL. If omitted, use --package.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("curated_dataset"),
        help="Directory where archive is downloaded and extracted.",
    )
    parser.add_argument(
        "--archive-name",
        type=str,
        help="Optional local archive file name.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download only, do not extract.",
    )
    parser.add_argument(
        "--sha256",
        type=str,
        help="Optional expected SHA-256 hash for archive integrity verification.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing archive/extracted folder.",
    )
    args = parser.parse_args()
    if not args.url and not args.package:
        parser.error("Provide --package {coco,yolo} or --url.")
    return args


def default_archive_name(url: str) -> str:
    path = Path(urlparse(url).path)
    name = path.name
    if name == "content" and len(path.parts) >= 2:
        # Zenodo API links end with /<filename>/content.
        return path.parts[-2]
    return name or "curated_dataset_download"


def extract_archive(archive_path: Path, extract_root: Path) -> Path:
    extract_dir = extract_root / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
        return extract_dir

    print(f"ERROR: Downloaded file is not a supported archive: {archive_path}")
    print("Use a direct Zenodo file download URL (zip/tar/tar.gz).")
    sys.exit(1)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_source(args: argparse.Namespace) -> tuple[str, str, str | None]:
    if args.url:
        url = args.url
        archive_name = args.archive_name or default_archive_name(url)
        expected_sha = args.sha256.strip().lower() if args.sha256 else None
        return url, archive_name, expected_sha

    package_meta = ZENODO_PACKAGES[args.package]
    url = package_meta["url"]
    archive_name = args.archive_name or package_meta["archive_name"]
    expected_sha = args.sha256.strip().lower() if args.sha256 else package_meta["sha256"]
    return url, archive_name, expected_sha


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    url, archive_name, expected_sha = resolve_source(args)
    archive_path = out_dir / archive_name

    if archive_path.exists():
        if not args.force:
            print(f"ERROR: Archive already exists: {archive_path}")
            print("Use --force to overwrite.")
            sys.exit(1)
        if archive_path.is_file():
            archive_path.unlink()
        else:
            shutil.rmtree(archive_path)

    if args.package:
        print(f"Zenodo DOI:  {ZENODO_DOI_URL}")
        print(f"Package:     {args.package}")
    print(f"Downloading: {url}")
    print(f"Saving to:   {archive_path}")
    urlretrieve(url, archive_path)
    print("Download complete.")

    if expected_sha:
        actual = file_sha256(archive_path)
        if actual != expected_sha:
            print("ERROR: SHA-256 mismatch.")
            print(f"  expected: {expected_sha}")
            print(f"  actual:   {actual}")
            sys.exit(1)
        print(f"SHA-256 verified: {actual}")

    if args.no_extract:
        return

    extracted = extract_archive(archive_path, out_dir)
    print(f"Extracted to: {extracted}")


if __name__ == "__main__":
    main()
