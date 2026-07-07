#!/usr/bin/env python3
"""
scripts/push_to_github.py
===========================
Formula1-AI  ·  Push project to GitHub as INFURNI09

Usage:
    python scripts/push_to_github.py --token YOUR_GITHUB_TOKEN

Steps:
    1. git init (if not already)
    2. git config user.name / user.email for INFURNI09
    3. git remote add origin https://github.com/INFURNI09/Formula1-AI.git
    4. git add --all
    5. git commit -m "feat: Formula1-AI — complete ML + ETL + API + Dashboard"
    6. git push -u origin main --force
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run(cmd: list[str], cwd: str = ".") -> tuple[int, str, str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def push_to_github(token: str, repo_url: str) -> None:
    _auth_url = repo_url.replace("https://", f"https://INFURNI09:{token}@")

    steps = [
        (["git", "init"],                                    "init repo"),
        (["git", "config", "user.name",  "INFURNI09"],      "set username"),
        (["git", "config", "user.email", "INFURNI09@users.noreply.github.com"],
                                                             "set email"),
        (["git", "checkout", "-B", "main"],                  "create main branch"),
        (["git", "remote", "remove", "origin"],              "remove old remote (ok if fails)"),
        (["git", "remote", "add", "origin", _auth_url],      "add remote"),
        (["git", "add", "--all"],                            "stage all files"),
        (["git", "commit", "-m",
          "feat: Formula1-AI — complete ML + ETL + API + XAI + Dashboard\n\n"
          "- ETL pipeline: Ergast/OpenF1/FastF1 → DuckDB (14 tables)\n"
          "- Feature engineering: 35+ motorsport features + dynamic Elo\n"
          "- Training: GradientBoosting/RF/XGB/LGB/CatBoost + Optuna HPO\n"
          "- Models: winner AUC=0.92, podium AUC=0.93, qualifying MAE=0.86\n"
          "- Monte Carlo: race + season simulation (1,000 iterations)\n"
          "- SHAP XAI: waterfall, summary, dependence plots + text explanations\n"
          "- Dash dashboard: 4 views (strategy, telemetry, championship, XAI)\n"
          "- FastAPI: 8 endpoints with Pydantic v2 request/response models"],
                                                             "commit"),
        (["git", "push", "-u", "origin", "main", "--force"], "push to GitHub"),
    ]

    for cmd, label in steps:
        code, out, err = run(cmd)
        status = "✅" if code == 0 else "⚠️ "
        print(f"  {status} {label}")
        if out:
            print(f"     {out[:120]}")
        if code != 0 and label not in ("remove old remote (ok if fails)",):
            if "nothing to commit" in err.lower():
                print("     (nothing to commit — already up to date)")
            elif code != 0:
                print(f"     stderr: {err[:200]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Formula1-AI to GitHub as INFURNI09")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument(
        "--repo",
        default="https://github.com/INFURNI09/Formula1-AI.git",
        help="Repository URL",
    )
    args = parser.parse_args()
    print("\n🏎 Formula1-AI — GitHub Push")
    print(f"   Repo: {args.repo}")
    print(f"   User: INFURNI09\n")
    push_to_github(args.token, args.repo)
    print("\n  🏁 Done. Check: https://github.com/INFURNI09/Formula1-AI")
