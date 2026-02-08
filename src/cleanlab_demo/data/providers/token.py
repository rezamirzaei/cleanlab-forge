"""
Token Classification Data Providers.

Provides data providers for token classification tasks (POS tagging, NER, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from cleanlab_demo.tasks.token.provider import TokenClassificationDataProvider
from cleanlab_demo.utils.download import download_file


class ConlluDataProvider(TokenClassificationDataProvider):
    """
    Generic provider for CoNLL-U format datasets.

    Parses standard CoNLL-U files for token classification.
    """

    def __init__(
        self,
        name: str,
        train_path: Path | str,
        dev_path: Path | str,
        token_col: int = 1,
        tag_col: int = 3,
    ) -> None:
        """
        Initialize the provider.

        Args:
            name: Dataset name for reporting
            train_path: Path to training file or URL
            dev_path: Path to dev file or URL
            token_col: Column index for tokens (0-based)
            tag_col: Column index for tags (0-based)
        """
        self._name = name
        self._train_path = Path(train_path) if isinstance(train_path, str) else train_path
        self._dev_path = Path(dev_path) if isinstance(dev_path, str) else dev_path
        self._token_col = token_col
        self._tag_col = tag_col

    @property
    def name(self) -> str:
        return self._name

    def _parse_conllu(
        self, path: Path, *, max_sentences: int | None = None
    ) -> tuple[list[list[str]], list[list[str]]]:
        """Parse a CoNLL-U file."""
        tokens_by_sent: list[list[str]] = []
        tags_by_sent: list[list[str]] = []
        cur_tokens: list[str] = []
        cur_tags: list[str] = []

        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                if cur_tokens:
                    tokens_by_sent.append(cur_tokens)
                    tags_by_sent.append(cur_tags)
                    cur_tokens, cur_tags = [], []
                    if max_sentences is not None and len(tokens_by_sent) >= max_sentences:
                        break
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) <= max(self._token_col, self._tag_col):
                continue
            tok_id = parts[0]
            if not tok_id.isdigit():
                continue
            cur_tokens.append(parts[self._token_col])
            cur_tags.append(parts[self._tag_col])

        if cur_tokens and (max_sentences is None or len(tokens_by_sent) < max_sentences):
            tokens_by_sent.append(cur_tokens)
            tags_by_sent.append(cur_tags)
        return tokens_by_sent, tags_by_sent

    def load(
        self, seed: int, max_train: int, max_dev: int, **kwargs: Any
    ) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]]]:
        """Load train/dev tokens and tags."""
        train_tokens, train_tags = self._parse_conllu(self._train_path, max_sentences=max_train)
        dev_tokens, dev_tags = self._parse_conllu(self._dev_path, max_sentences=max_dev)
        return train_tokens, train_tags, dev_tokens, dev_tags


class UDEnglishEWTProvider(TokenClassificationDataProvider):
    """
    Universal Dependencies English EWT dataset provider.

    POS tagging using the English Web Treebank.
    Source: Universal Dependencies
    """

    _UD_EWT_BASE: ClassVar[str] = (
        "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master"
    )
    _UD_FILES: ClassVar[dict[str, str]] = {
        "train": "en_ewt-ud-train.conllu",
        "dev": "en_ewt-ud-dev.conllu",
        "test": "en_ewt-ud-test.conllu",
    }

    def __init__(self, data_dir: Path | None = None) -> None:
        from cleanlab_demo.settings import settings
        self._data_dir = data_dir or (settings.data_dir / "ud_ewt")

    @property
    def name(self) -> str:
        return "UD_English-EWT (UPOS)"

    def _download_files(self) -> dict[str, Path]:
        """Download UD files if needed."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        out: dict[str, Path] = {}
        for split, fname in self._UD_FILES.items():
            url = f"{self._UD_EWT_BASE}/{fname}"
            out[split] = download_file(url, self._data_dir / fname)
        return out

    def _parse_conllu(
        self, path: Path, *, max_sentences: int | None = None
    ) -> tuple[list[list[str]], list[list[str]]]:
        """Parse a CoNLL-U file."""
        tokens_by_sent: list[list[str]] = []
        tags_by_sent: list[list[str]] = []
        cur_tokens: list[str] = []
        cur_tags: list[str] = []

        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                if cur_tokens:
                    tokens_by_sent.append(cur_tokens)
                    tags_by_sent.append(cur_tags)
                    cur_tokens, cur_tags = [], []
                    if max_sentences is not None and len(tokens_by_sent) >= max_sentences:
                        break
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            tok_id, form, upos = parts[0], parts[1], parts[3]
            if not tok_id.isdigit():
                continue
            cur_tokens.append(form)
            cur_tags.append(upos)

        if cur_tokens and (max_sentences is None or len(tokens_by_sent) < max_sentences):
            tokens_by_sent.append(cur_tokens)
            tags_by_sent.append(cur_tags)
        return tokens_by_sent, tags_by_sent

    def load(
        self, seed: int, max_train: int, max_dev: int, **kwargs: Any
    ) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]]]:
        """Load train/dev tokens and tags."""
        paths = self._download_files()
        train_tokens, train_tags = self._parse_conllu(paths["train"], max_sentences=max_train)
        dev_tokens, dev_tags = self._parse_conllu(paths["dev"], max_sentences=max_dev)
        return train_tokens, train_tags, dev_tokens, dev_tags
