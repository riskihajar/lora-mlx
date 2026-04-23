import os
from pathlib import Path

from openai import OpenAI

from .paths import REPO_ROOT


def load_dotenv(dotenv_path: Path | None = None) -> None:
    dotenv_path = dotenv_path or (REPO_ROOT / ".env")
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_openai_config() -> dict[str, str]:
    load_dotenv()
    required = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing OpenAI config in environment or .env: " + ", ".join(missing)
        )
    return {
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.environ["OPENAI_BASE_URL"],
        "model": os.environ["OPENAI_MODEL"],
    }


def build_client() -> tuple[OpenAI, str]:
    config = get_openai_config()
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
    return client, config["model"]


def validate_client() -> dict[str, object]:
    client, model = build_client()
    models = client.models.list()
    model_ids = [item.id for item in models.data]
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Reply with the single word: OK"}],
            }
        ],
    )
    return {
        "model": model,
        "model_found": model in model_ids,
        "models_count": len(model_ids),
        "test_output": response.output_text,
    }


def generate_text(system_prompt: str, user_prompt: str) -> str:
    client, model = build_client()
    response = client.responses.create(
        model=model,
        store=False,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    )
    return response.output_text
