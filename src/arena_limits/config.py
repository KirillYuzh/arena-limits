import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> None:
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env()

DB_PATH = os.environ.get("ARENA_DB", str(ROOT / "arena.duckdb"))
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-local")
DEFAULT_KAPPA = float(os.environ.get("ARENA_KAPPA", "0.1"))
DEFAULT_DELTA = float(os.environ.get("ARENA_DELTA", "0.05"))
DEFAULT_LGD = float(os.environ.get("ARENA_LGD", "1.0"))
DEFAULT_K = int(os.environ.get("ARENA_K", "50"))
