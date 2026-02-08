"""Tests for URL validation in download utility."""

from __future__ import annotations

from pathlib import Path

import pytest

from cleanlab_demo.utils.download import download_file


def test_download_file_rejects_file_scheme(tmp_path: Path) -> None:
    """download_file must reject file:// URLs to prevent local file access."""
    with pytest.raises(ValueError, match="not allowed"):
        download_file("file:///etc/passwd", tmp_path / "out.txt")


def test_download_file_rejects_ftp_scheme(tmp_path: Path) -> None:
    """download_file must reject ftp:// URLs."""
    with pytest.raises(ValueError, match="not allowed"):
        download_file("ftp://example.com/data.csv", tmp_path / "out.txt")


def test_download_file_rejects_empty_scheme(tmp_path: Path) -> None:
    """download_file must reject URLs without a scheme."""
    with pytest.raises(ValueError, match="not allowed"):
        download_file("/etc/passwd", tmp_path / "out.txt")
