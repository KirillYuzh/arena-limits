import os
from pathlib import Path

from litellm import completion

MODEL = "gigachat/GigaChat-2-Max"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def attempt(label: str, **kwargs) -> bool:
    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Ответь одним словом: works?"}],
            max_tokens=32,
            **kwargs,
        )
        content = (response.choices[0].message.content or "").strip()
        print(f"OK[{label}] model={response.model} reply={content[:100]}")
        return True
    except Exception as exc:
        print(f"FAIL[{label}] {type(exc).__name__}: {str(exc)[:220]}")
        return False


def main() -> int:
    load_env(Path(__file__).resolve().parents[1] / ".env")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    print(
        "cert env vars present:",
        sorted(
            k
            for k in os.environ
            if any(t in k.upper() for t in ("CERT", "CA_BUNDLE", "SSL"))
        ),
    )

    credentials = os.environ.get("GIGACHAT_CREDENTIALS", "").strip()
    if not credentials:
        print("GIGACHAT_CREDENTIALS не задан: проверь .env")
        return 1

    os.environ.setdefault("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

    if attempt("kwarg-ssl_verify", api_key=credentials, ssl_verify=False):
        return 0
    os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = "False"
    if attempt("env-verify-False", api_key=credentials):
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
