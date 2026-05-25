"""Pull cloud Pasal.id D2L eval results back to ``outputs/predictions/``.

Default repo: ``<username>/pasalid-d2l-eval-output``.
Default destination: ``outputs/predictions/pasalid_d2l_cloud/``.

Usage:

    python3 cloud/pull_pasalid_d2l_output.py
    python3 cloud/pull_pasalid_d2l_output.py --run 20260525_014941
    python3 cloud/pull_pasalid_d2l_output.py --repo myuser/some-repo --run latest
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, whoami

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "outputs/predictions/pasalid_d2l_cloud"


def _list_runs(repo_id: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    runs = sorted({p.split("/")[1] for p in files if p.startswith("run/")})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None)
    parser.add_argument(
        "--run",
        default="latest",
        help="run tag to pull (or 'latest', or 'all')",
    )
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    args = parser.parse_args()

    if args.repo is None:
        username = whoami()["name"]
        args.repo = f"{username}/pasalid-d2l-eval-output"

    runs = _list_runs(args.repo)
    if not runs:
        raise SystemExit(f"[pull] no run/* folders found in {args.repo}")
    print(f"[pull] available runs: {runs}")

    if args.run == "all":
        targets = runs
    elif args.run == "latest":
        targets = [runs[-1]]
    else:
        if args.run not in runs:
            raise SystemExit(f"[pull] run {args.run!r} not found")
        targets = [args.run]

    for tag in targets:
        dest = Path(args.dest) / tag
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(dest),
            allow_patterns=[f"run/{tag}/*"],
        )
        print(f"[pull] {tag} -> {dest}")

    print(f"[pull] done. dest: {args.dest}")


if __name__ == "__main__":
    main()
