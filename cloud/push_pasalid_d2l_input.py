"""Push the local d2l_eval split to a private HF dataset.

Default repo: ``<username>/pasalid-d2l-eval-input``.

Usage:

    python3 cloud/push_pasalid_d2l_input.py
    python3 cloud/push_pasalid_d2l_input.py --repo myuser/some-other-repo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, whoami

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = REPO_ROOT / "data/pasalid/d2l_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=str, default=None)
    parser.add_argument("--public", action="store_true", help="make the repo public")
    args = parser.parse_args()

    if args.repo is None:
        username = whoami()["name"]
        args.repo = f"{username}/pasalid-d2l-eval-input"

    api = HfApi()
    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    print(f"[push] repo: {args.repo} (private={not args.public})")

    files = [
        "test_seen.jsonl",
        "test_unseen.jsonl",
        "docs_seen.jsonl",
        "docs_unseen.jsonl",
        "manifest.json",
    ]
    for fname in files:
        path = LOCAL_DIR / fname
        if not path.exists():
            print(f"[push] skip missing: {path}")
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=fname,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"[push] uploaded {fname} ({path.stat().st_size} bytes)")

    print(f"[push] done. View at https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
