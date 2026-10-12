import hashlib
import subprocess


def prompt_hash(system: str) -> str:
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


def provider_of(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else "unknown"


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
