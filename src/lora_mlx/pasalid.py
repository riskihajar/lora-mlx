import argparse
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from .paths import DEFAULT_DATA_DIR, REPO_ROOT


PASAL_ID_BASE_URL = "https://pasal.id/api/v1"
DEFAULT_PASAL_ID_RAW_DIR = DEFAULT_DATA_DIR / "pasalid_raw"


def load_dotenv(dotenv_path: Path | None = None) -> None:
    dotenv_path = dotenv_path or (REPO_ROOT / ".env")
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_pasal_id_api_key(env_vars: tuple[str, ...] = ("PASAL_ID_API_KEY", "PASAL_ID_API_TOKEN")) -> str:
    load_dotenv()
    for env_var in env_vars:
        api_key = os.getenv(env_var)
        if api_key:
            return api_key
    expected = ", ".join(env_vars)
    raise RuntimeError(
        f"Missing Pasal.id API token. Add one of these env vars to the environment or .env file: {expected}."
    )


def get_pasal_id_base_url(env_var: str = "PASAL_ID_API_BASE_URL") -> str:
    load_dotenv()
    return os.getenv(env_var, PASAL_ID_BASE_URL)


class PasalIdClient:
    def __init__(self, api_key: str, base_url: str = PASAL_ID_BASE_URL, timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            query = parse.urlencode({k: v for k, v in params.items() if v is not None})
            url = f"{url}?{query}"

        req = request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "lora-mlx pasalid client",
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Pasal.id API error {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to reach Pasal.id API: {exc}") from exc

    def list_laws(
        self,
        law_type: str | None = None,
        year: int | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self._get(
            "/laws",
            {
                "type": law_type,
                "year": year,
                "status": status,
                "limit": limit,
                "offset": offset,
            },
        )

    def get_law_detail(self, frbr_uri: str) -> dict[str, Any]:
        normalized = frbr_uri.lstrip("/")
        return self._get(f"/laws/{normalized}")


def sanitize_frbr_uri(frbr_uri: str) -> str:
    return frbr_uri.strip("/").replace("/", "_")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def ingest_laws(
    client: PasalIdClient,
    output_dir: Path,
    law_type: str | None,
    year: int | None,
    status: str | None,
    limit: int,
    max_laws: int | None,
    sleep_seconds: float,
    include_details: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_laws: list[dict[str, Any]] = []
    offset = 0

    while True:
        page = client.list_laws(
            law_type=law_type,
            year=year,
            status=status,
            limit=limit,
            offset=offset,
        )
        laws = page.get("laws", [])
        if not laws:
            break

        for law in laws:
            all_laws.append(law)
            if max_laws is not None and len(all_laws) >= max_laws:
                break

        if max_laws is not None and len(all_laws) >= max_laws:
            break

        offset += len(laws)
        if len(laws) < limit:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    index_payload = {
        "fetched_at": int(time.time()),
        "filters": {
            "type": law_type,
            "year": year,
            "status": status,
            "limit": limit,
            "max_laws": max_laws,
        },
        "total_fetched": len(all_laws),
        "laws": all_laws,
    }
    write_json(output_dir / "laws_index.json", index_payload)

    if include_details:
        for law in all_laws:
            frbr_uri = law["frbr_uri"]
            detail = client.get_law_detail(frbr_uri)
            write_json(output_dir / f"{sanitize_frbr_uri(frbr_uri)}.json", detail)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return index_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Pasal.id law data into a local cache.")
    parser.add_argument("--type", default="UU", help="Law type filter, e.g. UU, PP, PERPRES")
    parser.add_argument("--year", type=int, default=None, help="Optional year filter")
    parser.add_argument(
        "--status",
        default="berlaku",
        help="Optional status filter, e.g. berlaku, dicabut, diubah",
    )
    parser.add_argument("--limit", type=int, default=50, help="Page size for /laws")
    parser.add_argument(
        "--max-laws",
        type=int,
        default=20,
        help="Maximum number of laws to fetch into the local cache",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PASAL_ID_RAW_DIR),
        help="Directory for local Pasal.id cache output",
    )
    parser.add_argument(
        "--skip-details",
        action="store_true",
        help="Only fetch the law index and skip per-law detail requests",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Delay between API requests to stay well within rate limits",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api_key = get_pasal_id_api_key()
    base_url = get_pasal_id_base_url()
    client = PasalIdClient(api_key=api_key, base_url=base_url)
    result = ingest_laws(
        client=client,
        output_dir=Path(args.output_dir),
        law_type=args.type,
        year=args.year,
        status=args.status,
        limit=args.limit,
        max_laws=args.max_laws,
        sleep_seconds=args.sleep_seconds,
        include_details=not args.skip_details,
    )
    print(f"fetched={result['total_fetched']}")
    print(f"output_dir={Path(args.output_dir)}")


if __name__ == "__main__":
    main()
