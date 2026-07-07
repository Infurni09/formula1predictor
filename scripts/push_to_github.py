#!/usr/bin/env python3
"""
scripts/push_to_github.py
==========================
Formula1-AI  ·  GitHub push utility
Usage: python scripts/push_to_github.py --token YOUR_PAT
       (or set GITHUB_TOKEN env var)

Repository: https://github.com/Infurni09/formula1predictor
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_URL   = "https://github.com/Infurni09/formula1predictor.git"
GITHUB_USER = "Infurni09"
DEFAULT_BRANCH = "main"


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and print it."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0 and check:
        print(f"  ❌ ERROR: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(result.returncode)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Push Formula1-AI to GitHub as Infurni09")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""),
                        help="GitHub Personal Access Token (or set GITHUB_TOKEN env var)")
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--message", default="feat: Formula1-AI — production ML pipeline v1.0")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: GitHub PAT required. Pass --token or set GITHUB_TOKEN env var.", file=sys.stderr)
        print("       Generate at: https://github.com/settings/tokens", file=sys.stderr)
        sys.exit(1)

    root = Path(__file__).parent.parent
    print(f"\n🏎  Formula1-AI — Pushing to GitHub")
    print(f"   Repo:   {REPO_URL}")
    print(f"   Branch: {args.branch}")
    print(f"   User:   {GITHUB_USER}\n")

    # 1. Init git if not already
    git_dir = root / ".git"
    if not git_dir.exists():
        run(["git", "init", "-b", DEFAULT_BRANCH], cwd=str(root))
    else:
        print("  ℹ️  Git repo already initialised")

    # 2. Configure user identity
    run(["git", "config", "user.email", "infurni09@github.com"], cwd=str(root))
    run(["git", "config", "user.name",  "Infurni09"],             cwd=str(root))

    # 3. Add .gitignore if missing
    gi = root / ".gitignore"
    if not gi.exists():
        gi.write_text("__pycache__/\n*.pyc\n.env\nmlruns/\n*.duckdb\n")

    # 4. Stage everything
    run(["git", "add", "-A"], cwd=str(root))

    # 5. Commit
    status = run(["git", "status", "--short"], cwd=str(root), check=False)
    if not status.stdout.strip():
        print("  ℹ️  Nothing to commit — working tree clean")
    else:
        run(["git", "commit", "-m", args.message], cwd=str(root))

    # 6. Set remote with token
    auth_url = REPO_URL.replace("https://", f"https://{GITHUB_USER}:{args.token}@")
    run(["git", "remote", "remove", "origin"], cwd=str(root), check=False)
    run(["git", "remote", "add", "origin", auth_url], cwd=str(root))

    # 7. Push
    run(["git", "branch", "-M", args.branch], cwd=str(root))
    run(["git", "push", "-u", "origin", args.branch, "--force"], cwd=str(root))

    print(f"\n  ✅ Successfully pushed to https://github.com/{GITHUB_USER}/formula1predictor")
    print(f"     View at: https://github.com/{GITHUB_USER}/formula1predictor")


if __name__ == "__main__":
    main()