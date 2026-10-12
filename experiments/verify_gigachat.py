import os
import sys
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
os.environ.setdefault("LITELLM_LOG", "CRITICAL")

CANDIDATES = [
    "GigaChat-2",
    "GigaChat-2-Pro",
    "GigaChat-2-Max",
]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def probe(model: str) -> bool:
    import litellm
    from litellm import completion

    litellm.suppress_debug_info = True
    try:
        response = completion(
            model=f"gigachat/{model}",
            messages=[{"role": "user", "content": "Ответь одним словом: works?"}],
            api_key=os.environ.get("GIGACHAT_CREDENTIALS"),
            ssl_verify=False,
            max_tokens=32,
        )
        content = (response.choices[0].message.content or "").strip()
        print(f"RESULT OK   {model:>20} -> {content[:50]}")
        return True
    except Exception as exc:
        print(f"RESULT FAIL {model:>20} -> {str(exc)[:100]}")
        return False


def main() -> int:
    load_env(Path(__file__).resolve().parents[1] / ".env")
    available = [model for model in (sys.argv[1:] or CANDIDATES) if probe(model)]
    print("AVAILABLE:", available)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
