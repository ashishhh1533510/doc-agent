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
from urllib.parse import urlsplit, unquote
from doc_agent.tools.language_detector import SUPPORTED_EXTENSIONS



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

def _parse_git_url(value: str) -> tuple[str, str | None, str | None, bool]:
    """Split a (possibly browser-copied) URL into its clonable parts.

    Returns (clone_url, branch, subpath, is_file):
      - clone_url: a URL git can actually clone (repo root, no ?query / #fragment)
      - branch:    branch/tag to clone, or None for the repo's default branch
      - subpath:   folder/file inside the repo to document, or None for the whole repo
      - is_file:   True when the URL pointed at a single file (a GitHub /blob/ link)

    Handles the pastes that currently 500:
      - root repo URL .............. https://github.com/psf/requests
      - URL with tracking params ... https://github.com/psf/requests?utm_source=x   (query dropped)
      - GitHub subfolder URL ....... https://github.com/o/r/tree/main/sub%20dir
      - GitHub single-file URL ..... https://github.com/o/r/blob/main/path/file.py
    """
    value = value.strip()

    # ssh / scp-style URLs ("git@host:org/repo.git") have no ?query or web "tree"
    # path, so there is nothing to normalize — clone them as-is.
    if value.startswith(("git@", "ssh://")):
        return value, None, None, False

    parts = urlsplit(value)                       # peels off ?query and #fragment for us
    segments = [s for s in parts.path.split("/") if s]

    # GitHub web URL pointing INSIDE a repo:
    #   <owner>/<repo>/(tree|blob)/<branch>/<subpath...>
    # NOTE: a branch name containing "/" (e.g. "feature/x") is ambiguous here — we take
    # the first segment after tree/blob as the branch. Works for main/master/tags.
    if (
        parts.netloc.endswith("github.com")
        and len(segments) >= 4
        and segments[2] in ("tree", "blob")
    ):
        owner, repo, kind, branch = segments[0], segments[1], segments[2], segments[3]
        subpath = "/".join(unquote(s) for s in segments[4:]) or None
        clone_url = f"{parts.scheme}://{parts.netloc}/{owner}/{repo}.git"
        return clone_url, unquote(branch), subpath, kind == "blob"

    # Plain repo URL. Rebuilding from scheme + netloc + path DROPS any ?utm_source=...
    # query and #fragment; rstrip removes a trailing slash. A ".git" suffix is preserved.
    clone_url = f"{parts.scheme}://{parts.netloc}{parts.path}".rstrip("/")
    return clone_url, None, None, False


def _parse_git_url(value: str) -> tuple[str, str | None, str | None, bool]:
    """Split a (possibly browser-copied) URL into its clonable parts.

    Returns (clone_url, branch, subpath, is_file):
      - clone_url: a URL git can actually clone (repo root, no ?query / #fragment)
      - branch:    branch/tag to clone, or None for the repo's default branch
      - subpath:   folder/file inside the repo to document, or None for the whole repo
      - is_file:   True when the URL pointed at a single file (a GitHub /blob/ link)

    Handles the pastes that otherwise 500:
      - root repo URL .............. https://github.com/psf/requests
      - URL with tracking params ... https://github.com/psf/requests?utm_source=x   (query dropped)
      - GitHub subfolder URL ....... https://github.com/o/r/tree/main/sub%20dir
      - GitHub single-file URL ..... https://github.com/o/r/blob/main/path/file.py
    """
    value = value.strip()

    # ssh / scp-style URLs ("git@host:org/repo.git") have no ?query or web "tree"
    # path, so there is nothing to normalize — clone them as-is.
    if value.startswith(("git@", "ssh://")):
        return value, None, None, False

    parts = urlsplit(value)                       # peels off ?query and #fragment for us
    segments = [s for s in parts.path.split("/") if s]

    # GitHub web URL pointing INSIDE a repo:
    #   <owner>/<repo>/(tree|blob)/<branch>/<subpath...>
    # NOTE: a branch name containing "/" (e.g. "feature/x") is ambiguous here — we take
    # the first segment after tree/blob as the branch. Works for main/master/tags.
    if (
        parts.netloc.endswith("github.com")
        and len(segments) >= 4
        and segments[2] in ("tree", "blob")
    ):
        owner, repo, kind, branch = segments[0], segments[1], segments[2], segments[3]
        subpath = "/".join(unquote(s) for s in segments[4:]) or None
        clone_url = f"{parts.scheme}://{parts.netloc}/{owner}/{repo}.git"
        return clone_url, unquote(branch), subpath, kind == "blob"

    # Plain repo URL. Rebuilding from scheme + netloc + path DROPS any ?utm_source=...
    # query and #fragment; rstrip removes a trailing slash. A ".git" suffix is preserved.
    clone_url = f"{parts.scheme}://{parts.netloc}{parts.path}".rstrip("/")
    return clone_url, None, None, False


@contextmanager
def resolve_input(project_path: str, token: str | None = None):
    """Yield a local directory path for the given input, cleaning up if needed.

    Usage:
        with resolve_input(project_path, token) as code_dir:
            facts = extract_from_directory(code_dir)
    """
    # 1) git URL -> clone to temp dir, then point at the requested subfolder (if any)
    if is_git_url(project_path):
        import git  # GitPython

        clone_url, branch, subpath, is_file = _parse_git_url(project_path)
        clone_url = _inject_token(
            clone_url,
            token or os.environ.get("GIT_TOKEN", ""),
        )

        tmp = tempfile.mkdtemp(prefix="doc_agent_clone_")
        try:
            clone_kwargs = {"depth": 1}
            if branch:
                clone_kwargs["branch"] = branch          # clone the branch the subfolder lives on
            git.Repo.clone_from(clone_url, tmp, **clone_kwargs)

            target = Path(tmp)
            if subpath:
                target = target / subpath
                if not target.exists():
                    raise FileNotFoundError(
                        f"Path '{subpath}' not found in the repo on branch "
                        f"'{branch or 'default'}'. Check the folder name and branch."
                    )

            if is_file:
                # A /blob/ link points at one file; document its folder (matches the
                # single-local-file behavior further down).
                yield str(target.parent)
            else:
                yield str(target)            # subfolder if subpath was given, else repo root
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
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Single-file input must be one of {supported}, got: {p.name}"
            )
        yield str(p.parent)
        return


    raise ValueError(f"Unsupported input: {project_path}")