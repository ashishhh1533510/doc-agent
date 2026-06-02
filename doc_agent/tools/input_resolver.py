"""
Input resolution: turn any supported input into a local directory of Python
code, then let the existing extractor work unchanged.

Supported inputs (project_path can be any of):
  - a local directory                         -> used as-is
  - a single local .py file                   -> its parent dir is used
  - a GitHub/Git URL (https or git@)          -> cloned to a temp dir

For private repos, a token can be supplied; it is injected into an https clone
URL. The temp clone is always cleaned up by the caller via the returned context.
"""

import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


_GIT_URL_RE = re.compile(
    r"""^(
        https?://[^\s]+?(?:\.git)?/?      |
        git@[^\s:]+:[^\s]+?(?:\.git)?/?   |
        ssh://[^\s]+
    )$""",
    re.VERBOSE,
)


def is_git_url(value: str) -> bool:
    """True if the string looks like a clonable git URL."""
    if not isinstance(value, str):
        return False
    v = value.strip()
    if v.startswith(("http://", "https://", "git@", "ssh://")):
        return bool(_GIT_URL_RE.match(v))
    return False


def _inject_token(url: str, token: str) -> str:
    """Insert an auth token into an https git URL for private-repo access."""
    if not token:
        return url
    if url.startswith("https://"):
        return "https://" + token + "@" + url[len("https://"):]
    return url


@contextmanager
def resolve_input(project_path: str, token: str | None = None):
    """Yield a local directory path for the given input, cleaning up if needed.

    Usage:
        with resolve_input(project_path, token) as code_dir:
            facts = extract_from_directory(code_dir)
    """
    # 1) git URL -> clone to temp dir
    if is_git_url(project_path):
        import git  # GitPython

        tmp = tempfile.mkdtemp(prefix="doc_agent_clone_")
        clone_url = _inject_token(
            project_path.strip(),
            token or os.environ.get("GIT_TOKEN", ""),
        )
        try:
            git.Repo.clone_from(clone_url, tmp, depth=1)
            yield tmp
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return

    # 2) local path
    p = Path(project_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {project_path}")

    if p.is_dir():
        yield str(p)
        return

    if p.is_file():
        if p.suffix != ".py":
            raise ValueError(f"Single-file input must be a .py file, got: {p.name}")
        yield str(p.parent)
        return

    raise ValueError(f"Unsupported input: {project_path}")