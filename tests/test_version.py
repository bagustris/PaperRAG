"""Test that version is consistent across pyproject.toml, __init__.py, and CHANGELOG.md."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _pyproject_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _init_version() -> str:
    from paperrag import __version__
    return __version__


def _changelog_version() -> str:
    line = (ROOT / "CHANGELOG.md").read_text().splitlines()[9]  # line 10 (0-indexed)
    m = re.search(r"\[(\d+\.\d+\.\d+)\]", line)
    assert m, f"Could not parse version from CHANGELOG.md line 10: {line!r}"
    return m.group(1)


def test_version_consistency():
    pyproject = _pyproject_version()
    init = _init_version()
    changelog = _changelog_version()

    assert pyproject == init, f"pyproject.toml ({pyproject}) != __init__.py ({init})"
    assert pyproject == changelog, f"pyproject.toml ({pyproject}) != CHANGELOG.md line 10 ({changelog})"
