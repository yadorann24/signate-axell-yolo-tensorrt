"""Audit a source-only repository without reading Git credentials or file history."""

import argparse
from pathlib import Path
import re


ROOT_FILES = {
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "pyproject.toml",
    ".gitignore",
    "requirements-cpu.lock",
}
SOURCE_DIRECTORIES = {"src/axell_export": ".py", "tests": ".py", "scripts": ".py", "docs": ".md"}
LOCAL_DIRECTORIES = {
    ".git",
    ".venv",
    ".venv-export",
    ".venv-training",
    "venv",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
}
SENSITIVE = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
)


def inspect(root: Path) -> list[str]:
    problems = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(p in LOCAL_DIRECTORIES or p.endswith(".egg-info") for p in relative.parts):
            continue
        name = relative.as_posix()
        if path.is_symlink():
            problems.append(f"Symlink: {name}")
            continue
        if path.is_dir():
            continue
        allowed = name in ROOT_FILES or name == ".github/workflows/ci.yml"
        allowed |= SOURCE_DIRECTORIES.get(relative.parent.as_posix()) == path.suffix
        if not allowed:
            problems.append(f"Not a publishable source path: {name}")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            problems.append(f"Non-text contents: {name}")
            continue
        if "\0" in content:
            problems.append(f"Binary contents: {name}")
        if any(pattern.search(content) for pattern in SENSITIVE):
            problems.append(f"Possible credential (value hidden): {name}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root
    if not root.is_dir():
        parser.error("--root must be a directory")
    problems = inspect(root)
    for problem in problems:
        print(problem)
    print("Publication source check: " + ("FAILED" if problems else "passed"))
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
