from pathlib import Path


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Use a path relative to the repository root.")
    return path
