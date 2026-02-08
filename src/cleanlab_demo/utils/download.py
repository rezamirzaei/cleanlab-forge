from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from cleanlab_demo.settings import logger
from cleanlab_demo.utils.fs import ensure_dir

_ALLOWED_SCHEMES = {"http", "https"}


class DownloadError(Exception):
    """Error during file download."""

    pass


def download_file(
    url: str,
    dest: Path,
    *,
    timeout_s: int = 120,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Path:
    """
    Download `url` to `dest` if not already present.

    Args:
        url: URL to download from (must use http or https scheme)
        dest: Destination file path
        timeout_s: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Path to the downloaded file

    Raises:
        DownloadError: If download fails after all retries
        ValueError: If the URL scheme is not allowed
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. Must be one of {_ALLOWED_SCHEMES}."
        )

    ensure_dir(dest.parent)
    if dest.exists() and dest.stat().st_size > 0:
        logger.debug(f"Using cached file: {dest}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
            with requests.get(url, stream=True, timeout=timeout_s) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                with tmp.open("wb") as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                    if total_size > 0:
                        logger.debug(f"Downloaded {downloaded:,} / {total_size:,} bytes")

            tmp.replace(dest)
            logger.info(f"Successfully downloaded to {dest}")
            return dest

        except requests.RequestException as e:
            last_error = e
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if tmp.exists():
                tmp.unlink()
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    raise DownloadError(f"Failed to download {url} after {max_retries} attempts: {last_error}")

