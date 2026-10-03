from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from .adapters import load_config, resolve_model_source


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate FCA-Bench local model configuration without loading model weights.")
    parser.add_argument("--config", required=True, help="JSON adapter configuration file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    backend = config.get("backend", "hf")
    print(f"Python: {sys.executable}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Backend: {backend}")

    if backend != "hf":
        print("Preflight OK: non-HF backend; model-path checks skipped.")
        return 0

    model_path = config.get("model_path")
    local_files_only = bool(config.get("local_files_only", True))
    try:
        resolved = resolve_model_source(model_path, local_files_only=local_files_only)
    except Exception as exc:
        print(f"Preflight FAILED: {type(exc).__name__}: {exc}")
        return 1

    print(f"Configured model path: {model_path}")
    print(f"Resolved model path: {resolved}")
    print(f"Local files only: {local_files_only}")

    local = Path(resolved)
    if local.exists():
        required = local / "config.json"
        print(f"Model config.json: {'OK' if required.is_file() else 'MISSING'}")
        if not required.is_file():
            print("Preflight FAILED: the resolved model directory does not contain config.json.")
            return 1

    required_packages = ["torch", "transformers"]
    if config.get("device_map") not in (None, "", "none"):
        required_packages.append("accelerate")
    missing = [name for name in required_packages if importlib.util.find_spec(name) is None]
    if missing:
        print("Preflight FAILED: missing Python packages: " + ", ".join(missing))
        return 1

    print("Python packages: " + ", ".join(f"{name}=OK" for name in required_packages))
    print("Preflight PASSED. Model weights were not loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
