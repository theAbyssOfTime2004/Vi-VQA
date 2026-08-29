#!/usr/bin/env python3
"""Check that everything the trainer's entry point imports is installed.

The Modal image lists the trainer's dependencies by hand. Nothing tied
that list to what the trainer actually imports, so a missing package
surfaced only when a real training run reached the import — three
minutes into a GPU container, one `ModuleNotFoundError` at a time.

This walks the import graph from `src/train/train_sft.py`, following the
trainer's own modules and collecting every third-party package it
reaches, then tries to import each one. Run it in the environment that
will do the training.

The graph matters more than a flat grep: `src/dataset/__init__.py`
imports the DPO and classification datasets too, so an SFT run needs
`trl` and `ujson` even though nothing in the SFT path uses them
directly. That is how both went missing.

Usage:
    python scripts/check_trainer_imports.py <path-to-cloned-trainer-repo>

Exits non-zero and names the missing packages, with the import chain
that reaches each one.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

#: Packages whose import name differs from what you `pip install`.
_PIP_NAME = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "qwen_vl_utils": "qwen-vl-utils",
    "cv2": "opencv-python",
}

ENTRYPOINT = Path("train") / "train_sft.py"


def _local_module(root: Path, dotted: str) -> Path | None:
    """The trainer's own file for a dotted name, or None if third-party."""
    parts = dotted.split(".")
    package = root.joinpath(*parts, "__init__.py")
    if package.is_file():
        return package
    module = root.joinpath(*parts).with_suffix(".py")
    return module if module.is_file() else None


def third_party_imports(root: Path, entrypoint: Path) -> dict[str, list[str]]:
    """Third-party top-level packages reachable from `entrypoint`.

    Returns:
        {package: the chain of imports that first reached it}
    """
    stdlib = set(sys.stdlib_module_names)
    visited: set[Path] = set()
    reached: dict[str, list[str]] = {}

    def walk(path: Path, chain: list[str]) -> None:
        if path in visited:
            return
        visited.add(path)

        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = path.parent.relative_to(root).as_posix().replace("/", ".")
                    targets = [f"{package}.{node.module}" if node.module else package]
                else:
                    targets = [node.module or ""]
            else:
                continue

            for target in targets:
                if not target:
                    continue
                local = _local_module(root, target)
                if local is not None:
                    walk(local, [*chain, target])
                    continue
                top = target.split(".")[0]
                if top not in stdlib and top not in reached:
                    reached[top] = [*chain, target]

    walk(root / entrypoint, [entrypoint.stem])
    return reached


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    root = Path(sys.argv[1]).resolve() / "src"
    entry = root / ENTRYPOINT
    if not entry.is_file():
        print(f"error: {entry} not found", file=sys.stderr)
        return 2

    reached = third_party_imports(root, ENTRYPOINT)
    missing = {
        name: chain
        for name, chain in reached.items()
        if importlib.util.find_spec(name) is None
    }

    for name in sorted(reached):
        status = "MISSING" if name in missing else "ok"
        print(f"  [{status:>7}] {name}")

    if missing:
        print(f"\nFAIL: {len(missing)} package(s) the trainer imports are not installed:\n")
        for name in sorted(missing):
            print(f"  {name}  (pip install {_PIP_NAME.get(name, name)})")
            print(f"      reached via {' -> '.join(missing[name])}")
        print("\nAdd them to the image in scripts/train_on_modal.py, or to the")
        print("local environment, before training.")
        return 1

    print(f"\nOK: all {len(reached)} packages the trainer imports are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
